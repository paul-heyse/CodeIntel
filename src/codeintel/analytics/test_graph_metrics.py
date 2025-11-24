"""Graph metrics over the test <-> function bipartite graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import networkx as nx
from networkx.algorithms import bipartite

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_test_function_bipartite
from codeintel.storage.gateway import StorageGateway


def _to_decimal(value: int) -> Decimal:
    return Decimal(value)


@dataclass(frozen=True)
class ProjectionBundle:
    """Projected graph metrics for one bipartite side."""

    degree: dict[tuple[str, object], int]
    weighted_degree: dict[tuple[str, object], float]
    clustering: dict[tuple[str, object], float]
    betweenness: dict[tuple[str, object], float]


@dataclass(frozen=True)
class BipartiteDegrees:
    """Degree mappings for the base bipartite graph."""

    degree: dict[tuple[str, object], int]
    weighted_degree: dict[tuple[str, object], float]
    test_dc: dict[tuple[str, object], float]
    func_dc: dict[tuple[str, object], float]


@dataclass(frozen=True)
class TestMetricsContext:
    """Shared context for computing test graph metrics."""

    repo: str
    commit: str
    now: datetime
    degrees: BipartiteDegrees
    risk_by_goid: dict[int, float]


def _projection_bundle(graph: nx.Graph, nodes: set[tuple[str, object]]) -> ProjectionBundle:
    proj = bipartite.weighted_projected_graph(graph, nodes)
    proj_deg = {node: int(deg) for node, deg in proj.degree(weight=None)}
    proj_wdeg = {node: float(deg) for node, deg in proj.degree(weight="weight")}
    if proj.number_of_nodes() > 0:
        clustering_val = nx.clustering(proj, weight="weight")
        proj_clustering = clustering_val if isinstance(clustering_val, dict) else {}
    else:
        proj_clustering = {}
    proj_bet = (
        nx.betweenness_centrality(proj, weight="weight", k=min(200, proj.number_of_nodes()))
        if proj.number_of_nodes() > 0
        else {}
    )
    return ProjectionBundle(
        degree=proj_deg,
        weighted_degree=proj_wdeg,
        clustering=proj_clustering,
        betweenness=proj_bet,
    )


def _bipartite_degrees(
    graph: nx.Graph, tests: set[tuple[str, object]], funcs: set[tuple[str, object]]
) -> BipartiteDegrees:
    degree_view = nx.degree(graph)
    degree = {node: int(deg) for node, deg in degree_view}
    weighted_view = nx.degree(graph, weight="weight")
    weighted_degree = {node: float(deg) for node, deg in weighted_view}
    test_dc = bipartite.degree_centrality(graph, funcs) if graph.number_of_nodes() > 0 else {}
    func_dc = bipartite.degree_centrality(graph, tests) if graph.number_of_nodes() > 0 else {}
    return BipartiteDegrees(
        degree=degree, weighted_degree=weighted_degree, test_dc=test_dc, func_dc=func_dc
    )


def _build_test_rows(
    graph: nx.Graph,
    tests: set[tuple[str, object]],
    ctx: TestMetricsContext,
) -> list[tuple[object, ...]]:
    if not tests:
        return []
    test_proj = _projection_bundle(graph, tests)
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
                float(ctx.degrees.test_dc.get(node, 0.0)),
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
    func_proj = _projection_bundle(graph, funcs)
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
                float(ctx.degrees.func_dc.get(node, 0.0)),
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
) -> None:
    """Populate test and function-side metrics derived from test coverage graphs."""
    con = gateway.con
    ensure_schema(con, "analytics.test_graph_metrics_tests")
    ensure_schema(con, "analytics.test_graph_metrics_functions")

    graph: nx.Graph = load_test_function_bipartite(gateway, repo, commit)
    now = datetime.now(UTC)

    tests = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    funcs = set(graph) - tests
    degrees = _bipartite_degrees(graph, tests, funcs)
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
