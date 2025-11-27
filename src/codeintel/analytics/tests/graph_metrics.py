"""Graph metrics over the test <-> function bipartite graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    BipartiteDegrees,
    GraphContext,
    bipartite_degrees,
    projection_metrics,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema


def _to_decimal(value: int) -> Decimal:
    return Decimal(value)


@dataclass(frozen=True)
class TestMetricsContext:
    """Shared context for computing test graph metrics."""

    repo: str
    commit: str
    now: datetime
    degrees: BipartiteDegrees
    risk_by_goid: dict[int, float]
    graph_ctx: GraphContext


def _build_test_rows(
    graph: nx.Graph,
    tests: set[tuple[str, object]],
    ctx: TestMetricsContext,
) -> list[tuple[object, ...]]:
    if not tests:
        return []
    test_proj = projection_metrics(
        graph,
        tests,
        ctx.graph_ctx,
    )
    rows: list[tuple[object, ...]] = []
    for node in tests:
        _, test_id = node
        risk_weighted = 0.0
        for neighbor in graph.neighbors(node):
            _, func_id = neighbor
            weight = float(graph[node][neighbor].get("weight", 0.0))
            risk_weighted += weight * ctx.risk_by_goid.get(int(func_id), 0.0)
        rows.append(
            (
                test_id,
                ctx.repo,
                ctx.commit,
                int(ctx.degrees.degree.get(node, 0)),
                float(ctx.degrees.weighted_degree.get(node, 0.0)),
                float(ctx.degrees.primary_degree_centrality.get(node, 0.0)),
                int(test_proj.degree.get(node, 0)),
                float(test_proj.weighted_degree.get(node, 0.0)),
                float(test_proj.clustering.get(node, 0.0)),
                float(test_proj.betweenness.get(node, 0.0)),
                risk_weighted,
                ctx.now,
            )
        )
    return rows


def _build_function_rows(
    graph: nx.Graph,
    funcs: set[tuple[str, object]],
    ctx: TestMetricsContext,
) -> list[tuple[object, ...]]:
    if not funcs:
        return []
    func_proj = projection_metrics(
        graph,
        funcs,
        ctx.graph_ctx,
    )
    rows: list[tuple[object, ...]] = []
    for node in funcs:
        _, goid = node
        goid_int = int(cast("int", goid))
        risk_score = ctx.risk_by_goid.get(goid_int, 0.0)
        tests_risk_weight = risk_score * float(ctx.degrees.weighted_degree.get(node, 0.0))
        rows.append(
            (
                _to_decimal(goid_int),
                ctx.repo,
                ctx.commit,
                int(ctx.degrees.degree.get(node, 0)),
                float(ctx.degrees.weighted_degree.get(node, 0.0)),
                float(ctx.degrees.secondary_degree_centrality.get(node, 0.0)),
                int(func_proj.degree.get(node, 0)),
                float(func_proj.weighted_degree.get(node, 0.0)),
                float(func_proj.clustering.get(node, 0.0)),
                float(func_proj.betweenness.get(node, 0.0)),
                tests_risk_weight,
                ctx.now,
            )
        )
    return rows


def compute_test_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate test and function-side metrics derived from test coverage graphs."""
    runtime_opts = runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    context = runtime_opts.context
    graph_ctx = runtime_opts.graph_ctx
    use_gpu = runtime.use_gpu if isinstance(runtime, GraphRuntime) else runtime_opts.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.test_graph_metrics_tests")
    ensure_schema(con, "analytics.test_graph_metrics_functions")

    if context is not None and (context.repo != repo or context.commit != commit):
        return

    if isinstance(runtime, GraphRuntime):
        graph = runtime.ensure_test_function_bipartite()
    else:
        engine: GraphEngine = runtime_opts.build_engine(gateway, repo, commit)
        graph = engine.test_function_bipartite()
    graph_ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=use_gpu,
    )
    if graph_ctx.use_gpu != use_gpu:
        graph_ctx = GraphContext(
            repo=graph_ctx.repo,
            commit=graph_ctx.commit,
            now=graph_ctx.now,
            betweenness_sample=graph_ctx.betweenness_sample,
            eigen_max_iter=graph_ctx.eigen_max_iter,
            seed=graph_ctx.seed,
            pagerank_weight=graph_ctx.pagerank_weight,
            betweenness_weight=graph_ctx.betweenness_weight,
            use_gpu=use_gpu,
        )
    now = graph_ctx.resolved_now()

    tests = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    funcs = set(graph) - tests
    degrees = bipartite_degrees(
        graph,
        tests,
        funcs,
        weight=graph_ctx.pagerank_weight,
    )
    risk_by_goid = {
        int(goid): float(score)
        for goid, score in con.execute(
            """
            SELECT function_goid_h128, risk_score
            FROM analytics.goid_risk_factors
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    }
    ctx = TestMetricsContext(
        repo=repo,
        commit=commit,
        now=now,
        degrees=degrees,
        risk_by_goid=risk_by_goid,
        graph_ctx=graph_ctx,
    )

    test_rows = _build_test_rows(graph, tests, ctx)
    func_rows = _build_function_rows(graph, funcs, ctx)

    con.execute(
        "DELETE FROM analytics.test_graph_metrics_tests WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.test_graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    if test_rows:
        con.executemany(
            """
            INSERT INTO analytics.test_graph_metrics_tests (
                test_id, repo, commit, degree, weighted_degree, degree_centrality,
                proj_degree, proj_weight, proj_clustering, proj_betweenness,
                risk_weighted_degree, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            test_rows,
        )
    if func_rows:
        con.executemany(
            """
            INSERT INTO analytics.test_graph_metrics_functions (
                function_goid_h128, repo, commit,
                tests_degree, tests_weighted_degree, tests_degree_centrality,
                proj_degree, proj_weight, proj_clustering, proj_betweenness,
                tests_risk_weighted_degree, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            func_rows,
        )
