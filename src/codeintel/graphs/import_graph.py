"""Construct module-level import graphs from LibCST parsing."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import libcst as cst
import networkx as nx

from codeintel.config import ImportGraphStepConfig
from codeintel.graphs.function_catalog import load_function_catalog
from codeintel.graphs.import_resolver import collect_import_edges
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import (
    ImportEdgeRow,
    ImportModuleRow,
    import_edge_to_tuple,
    import_module_to_tuple,
)
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _tarjan_scc(graph: dict[str, set[str]]) -> dict[str, int]:
    """
    Compute strongly connected components using NetworkX.

    Parameters
    ----------
    graph : dict[str, set[str]]
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
    """
    Compute topological layers for a DAG.

    Returns
    -------
    dict[str | int, int]
        Mapping of node -> layer depth.
    """
    layers: dict[str | int, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def components_and_layers(
    raw_edges: set[tuple[str, str]],
    modules: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute SCC membership and condensation layers from raw edges.

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
    condensation = nx.condensation(graph, scc=sccs) if graph.number_of_nodes() > 0 else nx.DiGraph()
    comp_layers = _dag_layers(condensation) if condensation.number_of_nodes() > 0 else {}
    layer_by_module = {node: comp_layers.get(scc_map.get(node, -1), 0) for node in graph.nodes}
    return scc_map, layer_by_module


def build_import_module_rows(
    repo: str,
    commit: str,
    modules: set[str],
    scc_map: dict[str, int],
    layers: dict[str, int],
) -> list[ImportModuleRow]:
    """
    Build rows for graph.import_modules from SCC and layering metadata.

    Returns
    -------
    list[ImportModuleRow]
        Sorted rows ready for insertion into graph.import_modules.
    """
    rows: list[ImportModuleRow] = []
    comp_sizes = Counter(scc_map.values())
    for module in sorted(modules):
        component_id = scc_map.get(module, -1)
        rows.append(
            ImportModuleRow(
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
    repo, commit = context
    module_rows: list[ImportModuleRow] = build_import_module_rows(
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
    repo, commit = context
    fan_out: dict[str, int] = defaultdict(int)
    fan_in: dict[str, int] = defaultdict(int)
    for src, dst in raw_edges:
        fan_out[src] += 1
        fan_in[dst] += 1

    rows: list[ImportEdgeRow] = []
    for src, dst in sorted(raw_edges):
        rows.append(
            ImportEdgeRow(
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
    """
    Populate `graph.import_graph_edges` from LibCST import analysis.

    Parameters
    ----------
    gateway : StorageGateway
        Gateway providing the DuckDB connection seeded with `core.modules`.
    cfg : ImportGraphStepConfig
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
        except Exception:
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
