"""Config bipartite/projection graph metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast

import duckdb
import networkx as nx

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_undirected,
    community_ids,
    log_empty_graph,
    log_projection_skipped,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_config_module_bipartite
from codeintel.storage.gateway import StorageGateway

MAX_BETWEENNESS_NODES = 1000
MAX_COMMUNITY_NODES = 5000


@dataclass(frozen=True)
class ProjectionMetrics:
    """Centrality bundle for a projected bipartite graph."""

    deg: dict[tuple[str, str], int]
    wdeg: dict[tuple[str, str], float]
    bet: dict[tuple[str, str], float]
    clo: dict[tuple[str, str], float]
    comm_map: dict[tuple[str, str], int]


def _clear_config_tables(gateway: StorageGateway, repo: str, commit: str) -> None:
    con = gateway.con
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_keys WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_key_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_module_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )


def _build_projection(graph: nx.Graph, nodes: set[tuple[str, str]]) -> nx.Graph:
    if not nodes:
        log_projection_skipped(
            "config_projection",
            "empty partition",
            nodes=0,
            graph_nodes=graph.number_of_nodes(),
        )
        return nx.Graph()
    proj = nx.Graph()
    proj.add_nodes_from(nodes)
    if len(nodes) <= 1:
        log_projection_skipped(
            "config_projection",
            "partition too small",
            nodes=len(nodes),
            graph_nodes=graph.number_of_nodes(),
        )
        return proj
    return nx.bipartite.weighted_projected_graph(graph, nodes)


def _projection_metrics(proj: nx.Graph, ctx: GraphContext) -> ProjectionMetrics:
    if proj.number_of_nodes() == 0:
        return ProjectionMetrics(deg={}, wdeg={}, bet={}, clo={}, comm_map={})
    centrality = centrality_undirected(proj, ctx, weight=ctx.pagerank_weight)
    deg = {node: int(sum(1 for _ in proj.neighbors(node))) for node in proj.nodes}
    wdeg = {
        node: float(sum(data.get("weight", 1.0) for _, _, data in proj.edges(node, data=True)))
        for node in proj.nodes
    }
    return ProjectionMetrics(
        deg=deg,
        wdeg=wdeg,
        bet=centrality.betweenness,
        clo=centrality.closeness,
        comm_map=community_ids(proj, weight=ctx.pagerank_weight),
    )


def _projection_rows(
    *,
    proj: nx.Graph,
    repo: str,
    commit: str,
    now: datetime,
    metrics: ProjectionMetrics,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_rows = [
        (
            repo,
            commit,
            node[1],
            metrics.deg.get(node, 0),
            metrics.wdeg.get(node, 0.0),
            metrics.bet.get(node, 0.0),
            metrics.clo.get(node, 0.0),
            metrics.comm_map.get(node),
            now,
        )
        for node in proj.nodes
    ]
    edge_rows = [
        (
            repo,
            commit,
            src[1],
            dst[1],
            float(data.get("weight", 1.0)),
            now,
        )
        for src, dst, data in proj.edges(data=True)
    ]
    return cast("list[tuple[object, ...]]", node_rows), cast("list[tuple[object, ...]]", edge_rows)


def _projection_payload(
    *,
    graph: nx.Graph,
    nodes: set[tuple[str, str]],
    snapshot: tuple[str, str],
    created_at: datetime,
    ctx: GraphContext,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    repo, commit = snapshot
    proj = _build_projection(graph, nodes)
    metrics = _projection_metrics(proj, ctx)
    return _projection_rows(
        proj=proj,
        repo=repo,
        commit=commit,
        now=created_at,
        metrics=metrics,
    )


def _persist_rows(con: duckdb.DuckDBPyConnection, sql: str, rows: list[tuple[object, ...]]) -> None:
    if rows:
        con.executemany(sql, rows)


def compute_config_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
) -> None:
    """Compute metrics for config keys/modules and their projections."""
    con = gateway.con
    ensure_schema(con, "analytics.config_graph_metrics_keys")
    ensure_schema(con, "analytics.config_graph_metrics_modules")
    ensure_schema(con, "analytics.config_projection_key_edges")
    ensure_schema(con, "analytics.config_projection_module_edges")

    if context is not None and (context.repo != repo or context.commit != commit):
        return

    graph = load_config_module_bipartite(gateway, repo, commit)
    if graph.number_of_nodes() == 0:
        log_empty_graph("config_module_bipartite", graph)
        _clear_config_tables(gateway, repo, commit)
        return
    created_at = graph_ctx.resolved_now() if graph_ctx is not None else datetime.now(UTC)
    ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=created_at,
        pagerank_weight="weight",
        betweenness_weight="weight",
    )
    if ctx.betweenness_sample > MAX_BETWEENNESS_NODES:
        ctx = replace(ctx, betweenness_sample=MAX_BETWEENNESS_NODES)

    keys = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    modules = set(graph) - keys
    if len(keys) == 0 or len(modules) == 0:
        log_projection_skipped(
            "config_projection",
            "missing partition",
            nodes=0,
            graph_nodes=graph.number_of_nodes(),
        )
        _clear_config_tables(gateway, repo, commit)
        return

    key_rows, key_edges = _projection_payload(
        graph=graph,
        nodes=keys,
        snapshot=(repo, commit),
        created_at=created_at,
        ctx=ctx,
    )
    module_rows, module_edges = _projection_payload(
        graph=graph,
        nodes=modules,
        snapshot=(repo, commit),
        created_at=created_at,
        ctx=ctx,
    )

    _clear_config_tables(gateway, repo, commit)

    _persist_rows(
        con,
        """
        INSERT INTO analytics.config_graph_metrics_keys (
            repo, commit, config_key, degree, weighted_degree,
            betweenness, closeness, community_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        key_rows,
    )
    _persist_rows(
        con,
        """
        INSERT INTO analytics.config_graph_metrics_modules (
            repo, commit, module, degree, weighted_degree,
            betweenness, closeness, community_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        module_rows,
    )
    _persist_rows(
        con,
        """
        INSERT INTO analytics.config_projection_key_edges (
            repo, commit, src_key, dst_key, weight, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        key_edges,
    )
    _persist_rows(
        con,
        """
        INSERT INTO analytics.config_projection_module_edges (
            repo, commit, src_module, dst_module, weight, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        module_edges,
    )
