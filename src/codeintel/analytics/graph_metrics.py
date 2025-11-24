"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural hotspots and coupling signals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb
import networkx as nx

from codeintel.config.models import GraphMetricsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.graphs.nx_views import load_call_graph, load_import_graph
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportNeighborStats:
    """Neighbor and edge count summaries for import graph edges."""

    in_neighbors: dict[str, set[str]]
    out_neighbors: dict[str, set[str]]
    in_counts: dict[str, int]
    out_counts: dict[str, int]


@dataclass(frozen=True)
class CentralityMetrics:
    """Computed centrality metrics for a directed graph."""

    pagerank: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]


@dataclass(frozen=True)
class ComponentMetadata:
    """Component membership and layering information for a directed graph."""

    ids: dict[Any, int]
    in_cycle: dict[Any, bool]
    layer: dict[Any, int]


def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """Populate analytics graph metrics tables for the provided repo/commit."""
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")
    _compute_function_graph_metrics(gateway, cfg)
    module_by_path = None
    if catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path
    _compute_module_graph_metrics(gateway, cfg, module_by_path=module_by_path)


def _dag_layers(graph: nx.DiGraph) -> dict[Any, int]:
    layers: dict[Any, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def _component_metadata(
    graph: nx.DiGraph,
) -> ComponentMetadata:
    if graph.number_of_nodes() == 0:
        return ComponentMetadata(ids={}, in_cycle={}, layer={})

    components = list(nx.strongly_connected_components(graph))
    comp_index = {node: idx for idx, comp in enumerate(components) for node in comp}
    cycle_member = {node: len(components[comp_index[node]]) > 1 for node in graph.nodes}

    condensation = nx.condensation(graph, components)
    comp_layers = _dag_layers(condensation)
    layer_by_node = {node: comp_layers.get(comp_index[node], 0) for node in graph.nodes}
    return ComponentMetadata(ids=comp_index, in_cycle=cycle_member, layer=layer_by_node)


def _component_metadata_from_import_table(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> ComponentMetadata | None:
    try:
        rows = gateway.con.execute(
            """
            SELECT module, scc_id, component_size, layer
            FROM graph.import_modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except duckdb.Error:
        return None
    if not rows:
        return None

    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    for module, scc_id, component_size, layer in rows:
        name = str(module)
        comp_id[name] = int(scc_id) if scc_id is not None else -1
        size = int(component_size) if component_size is not None else 0
        in_cycle[name] = size > 1
        if layer is not None:
            layer_by_module[name] = int(layer)
    return ComponentMetadata(ids=comp_id, in_cycle=in_cycle, layer=layer_by_module)


def _merge_component_metadata(
    graph_nodes: set[Any],
    computed: ComponentMetadata,
    cached: ComponentMetadata | None,
) -> ComponentMetadata:
    if cached is None:
        return computed
    ids = computed.ids.copy()
    in_cycle = computed.in_cycle.copy()
    layer = computed.layer.copy()
    for node in graph_nodes:
        if node in cached.ids:
            ids[node] = cached.ids[node]
            in_cycle[node] = cached.in_cycle.get(node, False)
            layer[node] = cached.layer.get(node, layer.get(node, 0))
    return ComponentMetadata(ids=ids, in_cycle=in_cycle, layer=layer)


def _centrality(graph: nx.DiGraph, max_betweenness_sample: int | None) -> CentralityMetrics:
    if graph.number_of_nodes() == 0:
        return CentralityMetrics(pagerank={}, betweenness={}, closeness={})

    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except (ImportError, ModuleNotFoundError, AttributeError):
        log.warning("SciPy unavailable; using zeroed PageRank scores")
        pagerank = dict.fromkeys(graph.nodes, 0.0)
    closeness = nx.closeness_centrality(graph)
    betweenness = (
        nx.betweenness_centrality(
            graph,
            k=max_betweenness_sample,
            seed=0,
            weight=None,
        )
        if max_betweenness_sample is not None and max_betweenness_sample < graph.number_of_nodes()
        else nx.betweenness_centrality(graph, weight=None)
    )
    return CentralityMetrics(pagerank=pagerank, betweenness=betweenness, closeness=closeness)


def _compute_function_graph_metrics(gateway: StorageGateway, cfg: GraphMetricsConfig) -> None:
    con = gateway.con
    graph = load_call_graph(gateway, cfg.repo, cfg.commit)

    neighbor_in: dict[int, set[int]] = defaultdict(set)
    neighbor_out: dict[int, set[int]] = defaultdict(set)
    in_edge_count: dict[int, int] = defaultdict(int)
    out_edge_count: dict[int, int] = defaultdict(int)
    for src, dst, data in graph.edges(data=True):
        weight = int(data.get("weight", 1))
        neighbor_out[int(src)].add(int(dst))
        neighbor_in[int(dst)].add(int(src))
        out_edge_count[int(src)] += weight
        in_edge_count[int(dst)] += weight

    centrality = _centrality(graph, cfg.max_betweenness_sample)
    component_meta = _component_metadata(graph)

    con.execute(
        "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(UTC)
    rows_to_insert: list[tuple[object, ...]] = [
        (
            cfg.repo,
            cfg.commit,
            node,
            len(neighbor_in.get(node, ())),
            len(neighbor_out.get(node, ())),
            in_edge_count.get(node, 0),
            out_edge_count.get(node, 0),
            centrality.pagerank.get(node),
            centrality.betweenness.get(node),
            centrality.closeness.get(node),
            component_meta.in_cycle.get(node, False),
            component_meta.ids.get(node),
            component_meta.layer.get(node),
            now,
        )
        for node in sorted(graph.nodes)
    ]

    if rows_to_insert:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_functions (
                repo, commit, function_goid_h128,
                call_fan_in, call_fan_out, call_in_degree, call_out_degree,
                call_pagerank, call_betweenness, call_closeness,
                call_cycle_member, call_cycle_id, call_layer, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        log.info(
            "graph_metrics_functions populated: %d rows for %s@%s",
            len(rows_to_insert),
            cfg.repo,
            cfg.commit,
        )


def _load_symbol_module_edges(
    gateway: StorageGateway,
    module_by_path: dict[str, str] | None,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    modules: set[str] = set()
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    if module_by_path is None:
        rows = gateway.con.execute(
            """
            SELECT m_use.module, m_def.module
            FROM graph.symbol_use_edges su
            LEFT JOIN core.modules m_def ON m_def.path = su.def_path
            LEFT JOIN core.modules m_use ON m_use.path = su.use_path
            WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
            """
        ).fetchall()

        for use_module, def_module in rows:
            src = str(use_module)
            dst = str(def_module)
            modules.update((src, dst))
            outbound[src].add(dst)
            inbound[dst].add(src)
        return modules, inbound, outbound

    path_rows = gateway.con.execute(
        "SELECT def_path, use_path FROM graph.symbol_use_edges"
    ).fetchall()
    for def_path, use_path in path_rows:
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None:
            continue
        modules.update((use_module, def_module))
        outbound[use_module].add(def_module)
        inbound[def_module].add(use_module)

    return modules, inbound, outbound


def _import_stats_from_graph(graph: nx.DiGraph) -> ImportNeighborStats:
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_counts: dict[str, int] = defaultdict(int)
    out_counts: dict[str, int] = defaultdict(int)
    for src, dst, data in graph.edges(data=True):
        weight = int(data.get("weight", 1))
        out_neighbors[src].add(dst)
        in_neighbors[dst].add(src)
        out_counts[src] += weight
        in_counts[dst] += weight
    return ImportNeighborStats(
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_counts=in_counts,
        out_counts=out_counts,
    )


def _compute_module_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsConfig,
    module_by_path: dict[str, str] | None,
) -> None:
    con = gateway.con
    graph = load_import_graph(gateway, cfg.repo, cfg.commit)
    symbol_modules, symbol_inbound, symbol_outbound = _load_symbol_module_edges(
        gateway, module_by_path
    )
    modules = set(graph.nodes) | symbol_modules
    module_rows = con.execute(
        "SELECT module FROM core.modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    for (module,) in module_rows:
        if module is not None:
            modules.add(str(module))
    if modules:
        graph.add_nodes_from(modules)

    import_stats = _import_stats_from_graph(graph)
    centrality = _centrality(graph, cfg.max_betweenness_sample)
    component_meta = _component_metadata(graph)
    cached_component_meta = _component_metadata_from_import_table(gateway, cfg.repo, cfg.commit)
    component_meta = _merge_component_metadata(modules, component_meta, cached_component_meta)

    con.execute(
        "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = datetime.now(UTC)
    rows_to_insert: list[tuple[object, ...]] = [
        (
            cfg.repo,
            cfg.commit,
            module,
            len(import_stats.in_neighbors.get(module, ())),
            len(import_stats.out_neighbors.get(module, ())),
            import_stats.in_counts.get(module, 0),
            import_stats.out_counts.get(module, 0),
            centrality.pagerank.get(module),
            centrality.betweenness.get(module),
            centrality.closeness.get(module),
            component_meta.in_cycle.get(module, False),
            component_meta.ids.get(module),
            component_meta.layer.get(module),
            len(symbol_inbound.get(module, ())),
            len(symbol_outbound.get(module, ())),
            now,
        )
        for module in sorted(modules)
    ]

    if rows_to_insert:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_modules (
                repo, commit, module,
                import_fan_in, import_fan_out, import_in_degree, import_out_degree,
                import_pagerank, import_betweenness, import_closeness,
                import_cycle_member, import_cycle_id, import_layer,
                symbol_fan_in, symbol_fan_out, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        log.info(
            "graph_metrics_modules populated: %d rows for %s@%s",
            len(rows_to_insert),
            cfg.repo,
            cfg.commit,
        )
