"""Config bipartite/projection graph metrics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    GraphContext,
    ProjectionMetrics,
    build_projection_graph,
    log_empty_graph,
    log_projection_skipped,
    projection_metrics,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import DuckDBConnection, StorageGateway

MAX_BETWEENNESS_NODES = 1000


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
            metrics.degree.get(node, 0),
            metrics.weighted_degree.get(node, 0.0),
            metrics.betweenness.get(node, 0.0),
            metrics.closeness.get(node, 0.0),
            metrics.community_id.get(node),
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
    created_at: datetime,
    ctx: GraphContext,
    label: str,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    repo, commit = ctx.repo, ctx.commit
    proj = build_projection_graph(
        graph,
        nodes,
        label=label,
    )
    metrics = projection_metrics(
        graph,
        nodes,
        ctx,
        projection=proj,
        label=label,
    )
    return _projection_rows(
        proj=proj,
        repo=repo,
        commit=commit,
        now=created_at,
        metrics=metrics,
    )


def _persist_rows(con: DuckDBConnection, sql: str, rows: list[tuple[object, ...]]) -> None:
    if rows:
        con.executemany(sql, rows)


def compute_config_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    """
    Compute metrics for config keys/modules and their projections.

    Parameters
    ----------
    gateway :
        Storage gateway used for reading graphs and writing metrics.
    repo : str
        Repository identifier anchoring the metrics.
    commit : str
        Commit hash anchoring the metrics snapshot.
    runtime : GraphRuntimeOptions | None
        Optional runtime options supplying cached graphs and backend selection.
    """
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    use_gpu = runtime.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.config_graph_metrics_keys")
    ensure_schema(con, "analytics.config_graph_metrics_modules")
    ensure_schema(con, "analytics.config_projection_key_edges")
    ensure_schema(con, "analytics.config_projection_module_edges")

    if context is not None and (context.repo != repo or context.commit != commit):
        return

    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    graph = engine.config_module_bipartite()
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
        use_gpu=use_gpu,
    )
    if ctx.betweenness_sample > MAX_BETWEENNESS_NODES:
        ctx = replace(ctx, betweenness_sample=MAX_BETWEENNESS_NODES)
    if ctx.use_gpu != use_gpu:
        ctx = replace(ctx, use_gpu=use_gpu)

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
        created_at=created_at,
        ctx=ctx,
        label="config_keys",
    )
    module_rows, module_edges = _projection_payload(
        graph=graph,
        nodes=modules,
        created_at=created_at,
        ctx=ctx,
        label="config_modules",
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
