"""Pure import analysis functions.

This module provides stateless functions for analyzing import relationships
without any database or file I/O.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class ImportEdge:
    """Represents an import edge.

    Attributes
    ----------
    src_module
        Importing module.
    dst_module
        Imported module.
    """

    src_module: str
    dst_module: str


@dataclass(frozen=True)
class ImportModuleRow:
    """Row data for graph.import_modules table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    module
        Module name.
    scc_id
        Strongly connected component ID.
    component_size
        Size of the SCC.
    layer
        Topological layer.
    cycle_group
        Cycle group ID.
    """

    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int


@dataclass(frozen=True)
class ImportEdgeRow:
    """Row data for graph.import_graph_edges table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    src_module
        Source module.
    dst_module
        Destination module.
    src_fan_out
        Fan-out of source module.
    dst_fan_in
        Fan-in of destination module.
    cycle_group
        Cycle group ID.
    module_layer
        Layer of the source module.
    """

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None


@dataclass(frozen=True)
class ImportAnalysisResult:
    """Result of import graph analysis.

    Attributes
    ----------
    edges
        Import edges.
    modules
        All module names.
    scc_map
        Module to SCC ID mapping.
    layer_map
        Module to layer mapping.
    """

    edges: tuple[ImportEdge, ...]
    modules: tuple[str, ...]
    scc_map: Mapping[str, int]
    layer_map: Mapping[str, int]


def collect_import_edges(
    module_name: str,
    imports: Sequence[tuple[str, tuple[str, ...]]],
) -> list[ImportEdge]:
    """Collect import edges from parsed imports.

    Parameters
    ----------
    module_name
        Name of the importing module.
    imports
        Sequence of (imported_module, names) tuples.

    Returns
    -------
    list[ImportEdge]
        Import edges.
    """
    edges: list[ImportEdge] = []
    for imported_module, _ in imports:
        if imported_module:
            edges.append(ImportEdge(src_module=module_name, dst_module=imported_module))
    return edges


def compute_scc(
    edges: Sequence[ImportEdge],
    modules: AbstractSet[str],
) -> dict[str, int]:
    """Compute strongly connected components.

    Parameters
    ----------
    edges
        Import edges.
    modules
        All module names.

    Returns
    -------
    dict[str, int]
        Module to SCC ID mapping.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    for edge in edges:
        graph.add_edge(edge.src_module, edge.dst_module)

    components = list(nx.strongly_connected_components(graph))
    return {node: idx for idx, comp in enumerate(components) for node in comp}


def compute_layers(
    edges: Sequence[ImportEdge],
    modules: AbstractSet[str],
    scc_map: Mapping[str, int],
) -> dict[str, int]:
    """Compute topological layers for modules.

    Parameters
    ----------
    edges
        Import edges.
    modules
        All module names.
    scc_map
        Module to SCC ID mapping.

    Returns
    -------
    dict[str, int]
        Module to layer mapping.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    for edge in edges:
        graph.add_edge(edge.src_module, edge.dst_module)

    sccs = list(nx.strongly_connected_components(graph))
    if graph.number_of_nodes() == 0:
        return {}

    condensation = nx.condensation(graph, scc=sccs)
    if condensation.number_of_nodes() == 0:
        return {}

    # Compute layers on condensation DAG
    comp_layers: dict[int, int] = {
        node: 0 for node in condensation.nodes if condensation.in_degree(node) == 0
    }
    for node in nx.topological_sort(condensation):
        base = comp_layers.get(node, 0)
        for succ in condensation.successors(node):
            comp_layers[succ] = max(comp_layers.get(succ, 0), base + 1)

    return {node: comp_layers.get(scc_map.get(node, -1), 0) for node in modules}


def analyze_imports(
    edges: Sequence[ImportEdge],
    modules: AbstractSet[str],
) -> ImportAnalysisResult:
    """Perform full import graph analysis.

    Parameters
    ----------
    edges
        Import edges.
    modules
        All module names.

    Returns
    -------
    ImportAnalysisResult
        Analysis results with SCC and layer mappings.
    """
    # Ensure all edge endpoints are in modules
    all_modules = set(modules)
    for edge in edges:
        all_modules.add(edge.src_module)
        all_modules.add(edge.dst_module)

    scc_map = compute_scc(edges, all_modules)
    layer_map = compute_layers(edges, all_modules, scc_map)

    return ImportAnalysisResult(
        edges=tuple(edges),
        modules=tuple(sorted(all_modules)),
        scc_map=scc_map,
        layer_map=layer_map,
    )


def build_import_module_rows(
    repo: str,
    commit: str,
    result: ImportAnalysisResult,
) -> list[ImportModuleRow]:
    """Build module rows from analysis result.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    result
        Import analysis result.

    Returns
    -------
    list[ImportModuleRow]
        Module rows for persistence.
    """
    comp_sizes = Counter(result.scc_map.values())
    return [
        ImportModuleRow(
            repo=repo,
            commit=commit,
            module=module,
            scc_id=result.scc_map.get(module, -1),
            component_size=comp_sizes.get(result.scc_map.get(module, -1), 1),
            layer=result.layer_map.get(module),
            cycle_group=result.scc_map.get(module, -1),
        )
        for module in result.modules
    ]


def build_import_edge_rows(
    repo: str,
    commit: str,
    result: ImportAnalysisResult,
) -> list[ImportEdgeRow]:
    """Build edge rows from analysis result.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    result
        Import analysis result.

    Returns
    -------
    list[ImportEdgeRow]
        Edge rows for persistence.
    """
    # Compute fan-in/fan-out
    fan_out: dict[str, int] = {}
    fan_in: dict[str, int] = {}
    for edge in result.edges:
        fan_out[edge.src_module] = fan_out.get(edge.src_module, 0) + 1
        fan_in[edge.dst_module] = fan_in.get(edge.dst_module, 0) + 1

    return [
        ImportEdgeRow(
            repo=repo,
            commit=commit,
            src_module=edge.src_module,
            dst_module=edge.dst_module,
            src_fan_out=fan_out.get(edge.src_module, 0),
            dst_fan_in=fan_in.get(edge.dst_module, 0),
            cycle_group=result.scc_map.get(edge.src_module, -1),
            module_layer=result.layer_map.get(edge.src_module),
        )
        for edge in result.edges
    ]


__all__ = [
    "ImportAnalysisResult",
    "ImportEdge",
    "ImportEdgeRow",
    "ImportModuleRow",
    "analyze_imports",
    "build_import_edge_rows",
    "build_import_module_rows",
    "collect_import_edges",
    "compute_layers",
    "compute_scc",
]
