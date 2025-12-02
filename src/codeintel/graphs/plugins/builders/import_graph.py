"""Import graph builder plugin using factory pattern.

This module provides the import graph builder as a graph plugin. All
orchestration logic for constructing module-level import graphs is here.

Uses resource injection pattern via ctx.require() to access storage.

Architecture notes:
- Pure computation functions are in graphs.compute.imports
- This plugin orchestrates file I/O and database persistence
- The compute layer is stateless and testable in isolation
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import libcst as cst
import networkx as nx

from codeintel.config import ImportGraphStepConfig
from codeintel.config.datasets import ImportEdgeRow as DatasetImportEdgeRow
from codeintel.config.datasets import ImportModuleRow as DatasetImportModuleRow
from codeintel.config.datasets import (
    import_edge_to_tuple,
    import_module_to_tuple,
)
from codeintel.graphs.catalog import load_function_catalog
from codeintel.graphs.compute.callgraph import collect_import_edges
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _tarjan_scc(graph: dict[str, set[str]]) -> dict[str, int]:
    """Compute strongly connected components using NetworkX.

    Parameters
    ----------
    graph
        Adjacency list mapping modules to their imported modules.

    Returns
    -------
    dict[str, int]
        Mapping of module name to component identifier.
    """
    nx_graph = nx.DiGraph()
    for src, targets in graph.items():
        for dst in targets:
            nx_graph.add_edge(src, dst)

    components = list(nx.strongly_connected_components(nx_graph))
    return {node: idx for idx, comp in enumerate(components) for node in comp}


def _dag_layers(graph: nx.DiGraph) -> dict[str | int, int]:
    """Compute topological layers for a DAG.

    Parameters
    ----------
    graph
        A directed acyclic graph.

    Returns
    -------
    dict[str | int, int]
        Mapping of node -> layer depth.
    """
    layers: dict[str | int, int] = {
        node: 0 for node in graph.nodes if graph.in_degree(node) == 0
    }
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def components_and_layers(
    raw_edges: set[tuple[str, str]],
    modules: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute SCC membership and condensation layers from raw edges.

    Parameters
    ----------
    raw_edges
        Set of (source, destination) import edge tuples.
    modules
        Set of module names.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        Mapping of module -> scc id, and module -> layer index.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    for src, dst in raw_edges:
        graph.add_edge(src, dst)
    sccs = list(nx.strongly_connected_components(graph))
    scc_map = {node: idx for idx, comp in enumerate(sccs) for node in comp}
    condensation = (
        nx.condensation(graph, scc=sccs) if graph.number_of_nodes() > 0 else nx.DiGraph()
    )
    comp_layers = _dag_layers(condensation) if condensation.number_of_nodes() > 0 else {}
    layer_by_module = {
        node: comp_layers.get(scc_map.get(node, -1), 0) for node in graph.nodes
    }
    return scc_map, layer_by_module


def build_import_module_rows(
    repo: str,
    commit: str,
    modules: set[str],
    scc_map: dict[str, int],
    layers: dict[str, int],
) -> list[DatasetImportModuleRow]:
    """Build rows for graph.import_modules from SCC and layering metadata.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    modules
        Set of module names.
    scc_map
        Module to SCC ID mapping.
    layers
        Module to layer mapping.

    Returns
    -------
    list[DatasetImportModuleRow]
        Sorted rows ready for insertion into graph.import_modules.
    """
    rows: list[DatasetImportModuleRow] = []
    comp_sizes = Counter(scc_map.values())
    for module in sorted(modules):
        component_id = scc_map.get(module, -1)
        rows.append(
            DatasetImportModuleRow(
                repo=repo,
                commit=commit,
                module=module,
                scc_id=component_id,
                component_size=comp_sizes.get(component_id, 1),
                layer=layers.get(module),
                cycle_group=component_id,
            )
        )
    return rows


def _persist_import_modules(
    gateway: StorageGateway,
    context: tuple[str, str],
    modules: set[str],
    scc: dict[str, int],
    layer_by_module: dict[str, int],
) -> int:
    """Persist import module rows to storage.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    context
        (repo, commit) tuple.
    modules
        Set of module names.
    scc
        Module to SCC ID mapping.
    layer_by_module
        Module to layer mapping.

    Returns
    -------
    int
        Number of rows persisted.
    """
    repo, commit = context
    module_rows: list[DatasetImportModuleRow] = build_import_module_rows(
        repo,
        commit,
        modules,
        scc,
        layer_by_module,
    )
    run_batch(
        gateway,
        "graph.import_modules",
        [import_module_to_tuple(row) for row in module_rows],
        delete_params=[repo, commit],
        scope="import_modules",
    )
    return len(module_rows)


def _persist_import_edges(
    gateway: StorageGateway,
    context: tuple[str, str],
    raw_edges: set[tuple[str, str]],
    scc: dict[str, int],
    layer_by_module: dict[str, int],
) -> int:
    """Persist import edge rows to storage.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    context
        (repo, commit) tuple.
    raw_edges
        Set of (src_module, dst_module) edge tuples.
    scc
        Module to SCC ID mapping.
    layer_by_module
        Module to layer mapping.

    Returns
    -------
    int
        Number of rows persisted.
    """
    repo, commit = context
    fan_out: dict[str, int] = defaultdict(int)
    fan_in: dict[str, int] = defaultdict(int)
    for src, dst in raw_edges:
        fan_out[src] += 1
        fan_in[dst] += 1

    rows: list[DatasetImportEdgeRow] = []
    for src, dst in sorted(raw_edges):
        rows.append(
            DatasetImportEdgeRow(
                repo=repo,
                commit=commit,
                src_module=src,
                dst_module=dst,
                src_fan_out=fan_out.get(src, 0),
                dst_fan_in=fan_in.get(dst, 0),
                cycle_group=scc.get(src, -1),
                module_layer=layer_by_module.get(src),
            )
        )

    run_batch(
        gateway,
        "graph.import_graph_edges",
        [import_edge_to_tuple(row) for row in rows],
        delete_params=[repo, commit],
        scope="import_graph_edges",
    )
    return len(rows)


def build_import_graph(gateway: StorageGateway, cfg: ImportGraphStepConfig) -> None:
    """Populate graph.import_graph_edges from LibCST import analysis.

    Parameters
    ----------
    gateway
        Gateway providing the DuckDB connection seeded with core.modules.
    cfg
        Repository context and filesystem root.

    Notes
    -----
    The collector resolves relative imports conservatively to the current
    package. Strongly connected components are computed to identify cycles.
    """
    repo_root = cfg.repo_root.resolve()

    catalog = load_function_catalog(gateway, repo=cfg.repo, commit=cfg.commit)
    module_map = catalog.module_by_path
    if not module_map:
        log.info("No modules found in catalog; skipping import graph.")
        return

    # Collect raw edges
    raw_edges: set[tuple[str, str]] = set()
    for rel_path, module_name in module_map.items():
        file_path = repo_root / rel_path

        try:
            source = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("File missing for import graph: %s", file_path)
            continue

        try:
            module = cst.parse_module(source)
        except (cst.ParserSyntaxError, ValueError):
            log.exception("Failed to parse %s for import graph", file_path)
            continue

        raw_edges.update(collect_import_edges(module_name, module))

    # Build fan-out / fan-in / SCCs
    modules = set(module_map.values())
    for src, dst in raw_edges:
        modules.add(src)
        modules.add(dst)

    scc, layer_by_module = components_and_layers(raw_edges, modules)

    context = (cfg.repo, cfg.commit)
    module_count = _persist_import_modules(gateway, context, modules, scc, layer_by_module)
    edge_count = _persist_import_edges(gateway, context, raw_edges, scc, layer_by_module)

    log.info(
        "Import graph build complete for repo=%s commit=%s: %d edges, %d modules",
        cfg.repo,
        cfg.commit,
        edge_count,
        module_count,
    )


def _build_import_graph(ctx: GraphExecutionContext) -> ComputationResult:
    """Build module-level import graphs from LibCST parsing.

    Uses resource injection to access storage.

    Returns
    -------
    ComputationResult
        Success result after constructing import graph tables.
    """
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    cfg = ImportGraphStepConfig(snapshot=ctx.snapshot)
    build_import_graph(gateway, cfg)
    return ComputationResult.ok()


import_graph_builder_plugin = make_builder_plugin(
    name="import_graph_builder",
    computation=_build_import_graph,
    stage="structure",
    produces_graphs=(GraphKind.IMPORT_GRAPH,),
    depends_on=(),
    provides=("import_graph",),
    produces_tables=("graph.import_modules", "graph.import_edges"),
)


def get_import_graph_builder_plugin() -> GraphPluginProtocol:
    """Return the import graph builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured import graph builder plugin.
    """
    return import_graph_builder_plugin


__all__ = [
    "build_import_graph",
    "build_import_module_rows",
    "components_and_layers",
    "get_import_graph_builder_plugin",
    "import_graph_builder_plugin",
]
