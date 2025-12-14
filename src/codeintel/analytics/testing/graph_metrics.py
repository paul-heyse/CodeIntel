"""Graph metrics over the test <-> function bipartite graph.

Column definitions and internal helper functions for test graph metrics.

The pure compute functions are available in ``codeintel.analytics.testing.compute``:
- ``compute_test_graph_metrics_pure`` returns ``TestGraphMetricsResult``

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.test_graph_metrics``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from codeintel.analytics.compute.graphs import (
    projection_metrics,
)

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.analytics.compute.graphs import (
        BipartiteDegrees,
    )
    from codeintel.graphs.runtime.context import GraphContext

TEST_GRAPH_METRICS_TESTS_COLS = [
    "test_id",
    "repo",
    "commit",
    "degree",
    "weighted_degree",
    "degree_centrality",
    "proj_degree",
    "proj_weight",
    "proj_clustering",
    "proj_betweenness",
    "risk_weighted_degree",
    "created_at",
]
TEST_GRAPH_METRICS_FUNCTIONS_COLS = [
    "function_goid_h128",
    "repo",
    "commit",
    "tests_degree",
    "tests_weighted_degree",
    "tests_degree_centrality",
    "proj_degree",
    "proj_weight",
    "proj_clustering",
    "proj_betweenness",
    "tests_risk_weighted_degree",
    "created_at",
]


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
