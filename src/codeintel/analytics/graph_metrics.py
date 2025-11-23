"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural hotspots and coupling signals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import networkx as nx

from codeintel.config.models import GraphMetricsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)


def compute_graph_metrics(con: duckdb.DuckDBPyConnection, cfg: GraphMetricsConfig) -> None:
    """Populate analytics graph metrics tables for the provided repo/commit."""
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")
    _compute_function_graph_metrics(con, cfg)
    _compute_module_graph_metrics(con, cfg)


def _normalize_goid(raw: object) -> int:
    return int(raw)


def _dag_layers(graph: nx.DiGraph) -> dict[int, int]:
    layers: dict[int, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def _component_metadata(
    graph: nx.DiGraph,
) -> tuple[dict[int, int], dict[int, bool], dict[int, int]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, {}

    components = list(nx.strongly_connected_components(graph))
    comp_index = {node: idx for idx, comp in enumerate(components) for node in comp}
    cycle_member = {node: len(components[comp_index[node]]) > 1 for node in graph.nodes}

    condensation = nx.condensation(graph, components)
    comp_layers = _dag_layers(condensation)
    layer_by_node = {node: comp_layers.get(comp_index[node], 0) for node in graph.nodes}
    return comp_index, cycle_member, layer_by_node


def _centrality(
    graph: nx.DiGraph, max_betweenness_sample: int | None
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, {}

    pagerank = nx.pagerank(graph, weight="weight")
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
    return pagerank, betweenness, closeness


def _load_call_graph(
    con: duckdb.DuckDBPyConnection,
) -> tuple[set[int], dict[tuple[int, int], int]]:
    nodes: set[int] = set()
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)

    rows = con.execute(
        """
        SELECT caller_goid_h128, callee_goid_h128
        FROM graph.call_graph_edges
        WHERE callee_goid_h128 IS NOT NULL
        """
    ).fetchall()
    for caller_raw, callee_raw in rows:
        caller = _normalize_goid(caller_raw)
        callee = _normalize_goid(callee_raw)
        nodes.update((caller, callee))
        edge_counts[caller, callee] += 1

    node_rows = con.execute("SELECT goid_h128 FROM graph.call_graph_nodes").fetchall()
    for (node_raw,) in node_rows:
        if node_raw is not None:
            nodes.add(_normalize_goid(node_raw))

    return nodes, edge_counts


def _build_directed_graph(
    nodes: Iterable[int], edge_counts: dict[tuple[int, int], int]
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (src, dst), weight in edge_counts.items():
        graph.add_edge(src, dst, weight=weight)
    return graph


def _compute_function_graph_metrics(
    con: duckdb.DuckDBPyConnection, cfg: GraphMetricsConfig
) -> None:
    nodes, edge_counts = _load_call_graph(con)
    graph = _build_directed_graph(nodes, edge_counts)

    neighbor_in: dict[int, set[int]] = defaultdict(set)
    neighbor_out: dict[int, set[int]] = defaultdict(set)
    in_edge_count: dict[int, int] = defaultdict(int)
    out_edge_count: dict[int, int] = defaultdict(int)
    for (src, dst), count in edge_counts.items():
        neighbor_out[src].add(dst)
        neighbor_in[dst].add(src)
        out_edge_count[src] += count
        in_edge_count[dst] += count

    pagerank, betweenness, closeness = _centrality(graph, cfg.max_betweenness_sample)
    cycle_id, cycle_member, layer_by_node = _component_metadata(graph)

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
            pagerank.get(node),
            betweenness.get(node),
            closeness.get(node),
            cycle_member.get(node, False),
            cycle_id.get(node),
            layer_by_node.get(node),
            now,
        )
        for node in sorted(nodes)
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


def _load_import_edges(
    con: duckdb.DuckDBPyConnection,
) -> tuple[set[str], dict[tuple[str, str], int]]:
    modules: set[str] = set()
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    rows = con.execute("SELECT src_module, dst_module FROM graph.import_graph_edges").fetchall()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        src_module = str(src)
        dst_module = str(dst)
        modules.update((src_module, dst_module))
        edge_counts[src_module, dst_module] += 1

    module_rows = con.execute("SELECT module FROM core.modules").fetchall()
    for (module,) in module_rows:
        if module is not None:
            modules.add(str(module))

    return modules, edge_counts


def _load_symbol_module_edges(
    con: duckdb.DuckDBPyConnection,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    modules: set[str] = set()
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    rows = con.execute(
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


@dataclass(frozen=True)
class ImportNeighborStats:
    """Neighbor and edge count summaries for import graph edges."""

    in_neighbors: dict[str, set[str]]
    out_neighbors: dict[str, set[str]]
    in_counts: dict[str, int]
    out_counts: dict[str, int]


def _summarize_import_edges(edge_counts: dict[tuple[str, str], int]) -> ImportNeighborStats:
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_counts: dict[str, int] = defaultdict(int)
    out_counts: dict[str, int] = defaultdict(int)
    for (src, dst), count in edge_counts.items():
        out_neighbors[src].add(dst)
        in_neighbors[dst].add(src)
        out_counts[src] += count
        in_counts[dst] += count
    return ImportNeighborStats(
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_counts=in_counts,
        out_counts=out_counts,
    )


def _compute_module_graph_metrics(con: duckdb.DuckDBPyConnection, cfg: GraphMetricsConfig) -> None:
    import_modules, import_edges = _load_import_edges(con)
    symbol_modules, symbol_inbound, symbol_outbound = _load_symbol_module_edges(con)
    modules = import_modules | symbol_modules

    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    for (src, dst), weight in import_edges.items():
        graph.add_edge(src, dst, weight=weight)

    import_stats = _summarize_import_edges(import_edges)
    centrality = _centrality(graph, cfg.max_betweenness_sample)
    component_meta = _component_metadata(graph)

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
            centrality[0].get(module),
            centrality[1].get(module),
            centrality[2].get(module),
            component_meta[1].get(module, False),
            component_meta[0].get(module),
            component_meta[2].get(module),
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
