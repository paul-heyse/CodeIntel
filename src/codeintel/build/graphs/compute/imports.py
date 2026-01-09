"""Pure import analysis functions.

This module provides stateless functions for analyzing import relationships
without any database or file I/O.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import rustworkx as rx

from codeintel.build.graphs.rx.components import component_membership_by_id, sort_components
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.payloads import encode_node_payload
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.data_models.rows import ImportEdgeRow, ImportModuleRow

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet


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


def _build_import_store(
    edges: Sequence[ImportEdge],
    modules: AbstractSet[str],
) -> RxGraphStore:
    store = RxGraphStore.directed(node_hint=len(modules), edge_hint=len(edges))
    for edge in sorted(
        edges,
        key=lambda item: (stable_key(item.src_module), stable_key(item.dst_module)),
    ):
        store.add_weighted_edge(edge.src_module, edge.dst_module, weight=1.0)
    missing = [module for module in modules if module not in store.id_to_index]
    if not missing:
        return store
    isolate_graph = rx.PyDiGraph(multigraph=False)
    for module in sorted(missing, key=stable_key):
        isolate_graph.add_node(encode_node_payload(module, {}))
    directed_graph = cast("rx.PyDiGraph", store.graph)
    merged = rx.union(directed_graph, isolate_graph, merge_nodes=True, merge_edges=True)
    return RxGraphStore.from_rx_graph(
        merged,
        weight_policy=store.weight_policy,
        numeric_policy=store.numeric_policy,
    )


def _condensation_components(
    store: RxGraphStore,
) -> tuple[rx.PyDiGraph, list[set[int]], dict[int, int]] | None:
    condensed = cast("rx.PyDiGraph", rx.condensation(cast("rx.PyDiGraph", store.graph)))
    node_map = condensed.attrs.get("node_map")
    if not isinstance(node_map, Sequence) or isinstance(
        node_map, (str, bytes, bytearray, memoryview)
    ):
        return None
    components_by_id: dict[int, set[int]] = {}
    for node_idx, comp_id in enumerate(node_map):
        if isinstance(comp_id, int):
            components_by_id.setdefault(comp_id, set()).add(node_idx)
    if not components_by_id:
        return condensed, [], {}
    sorted_components = sort_components(store, components_by_id.values())
    component_lookup = {
        frozenset(component): old_id for old_id, component in components_by_id.items()
    }
    old_to_new: dict[int, int] = {}
    for new_id, component in enumerate(sorted_components):
        old_id = component_lookup.get(frozenset(component))
        if old_id is not None:
            old_to_new[old_id] = new_id
    return condensed, sorted_components, old_to_new


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
    store = _build_import_store(edges, modules)

    if store.graph.num_nodes() == 0:
        return {}

    condensation_data = _condensation_components(store)
    if condensation_data is None:
        return {}
    _condensed_graph, components, _old_to_new = condensation_data
    if not components:
        return {}
    return {
        str(node_id): comp_id
        for node_id, comp_id in component_membership_by_id(store, components).items()
    }


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
    if not modules:
        return {}
    store = _build_import_store(edges, modules)
    if store.graph.num_nodes() == 0:
        return {}
    if not scc_map:
        return {}
    condensation_data = _condensation_components(store)
    if condensation_data is None:
        return {}
    condensed, _components, old_to_new = condensation_data
    comp_layers: dict[int, int] = {}
    for layer, generation in enumerate(rx.topological_generations(condensed)):
        for comp_id in generation:
            mapped = old_to_new.get(comp_id)
            if mapped is not None:
                comp_layers[mapped] = layer
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
    all_modules = set(modules)
    for edge in edges:
        all_modules.add(edge.src_module)
        all_modules.add(edge.dst_module)

    scc_map = compute_scc(edges, all_modules)
    layer_map = compute_layers(edges, all_modules, scc_map)

    return ImportAnalysisResult(
        edges=tuple(edges),
        modules=tuple(sorted(all_modules, key=stable_key)),
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
