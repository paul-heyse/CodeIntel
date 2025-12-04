"""Smoke tests for graph metrics aggregation."""

from __future__ import annotations

from codeintel.analytics.graphs import compute_graph_metrics
from codeintel.config import ConfigBuilder
from tests._helpers import TestContext

# Expected row count constant
EXPECTED_GRAPH_METRICS_ROWS = 4


def test_compute_graph_metrics_with_seeded_data(graph_ctx: TestContext) -> None:
    """Verify graph metrics populate with pre-seeded graph data.

    Uses GRAPH_PACK which seeds call graph nodes and edges for 4 GOIDs.
    The compute_graph_metrics function should produce metrics for each.
    """
    cfg = ConfigBuilder.from_snapshot(
        repo=graph_ctx.repo,
        commit=graph_ctx.commit,
        repo_root=graph_ctx.repo_root,
    ).graph_metrics()

    compute_graph_metrics(graph_ctx.gateway, cfg)

    row_count = graph_ctx.query_count(
        "analytics.graph_metrics_functions",
        f"repo = '{graph_ctx.repo}' AND commit = '{graph_ctx.commit}'",
    )

    assert row_count == EXPECTED_GRAPH_METRICS_ROWS, (
        f"Expected {EXPECTED_GRAPH_METRICS_ROWS} function metric rows, got {row_count}"
    )
