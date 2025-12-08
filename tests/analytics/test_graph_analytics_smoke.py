"""Smoke tests for graph analytics computations.

This module consolidates smoke tests for graph stats and metrics
to verify basic functionality of the analytics graph layer.
"""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs import compute_graph_metrics, compute_graph_stats
from codeintel.config import ConfigBuilder, SnapshotInit
from tests._helpers import TestContext
from tests._helpers.assertions import expect_equal
from tests._helpers.seeds import GRAPH_PACK

# Expected row count constant for graph metrics
EXPECTED_GRAPH_METRICS_ROWS = 4


def test_graph_stats_records_counts_for_basic_graphs(test_ctx: TestContext) -> None:
    """Ensure graph_stats has entries after minimal graph ingestion."""
    # Apply graph seeds - provides call graph nodes/edges and import graph edges
    test_ctx.require(GRAPH_PACK)

    compute_graph_stats(test_ctx.gateway, repo=test_ctx.repo, commit=test_ctx.commit)

    rows = test_ctx.query(
        "SELECT graph_name, node_count FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [test_ctx.repo, test_ctx.commit],
    )
    if not rows:
        pytest.fail("Expected graph_stats to contain rows for ingested graphs")


def test_compute_graph_metrics_with_seeded_data(graph_ctx: TestContext) -> None:
    """Verify graph metrics populate with pre-seeded graph data.

    Uses GRAPH_PACK which seeds call graph nodes and edges for 4 GOIDs.
    The compute_graph_metrics function should produce metrics for each.
    """
    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(
            repo=graph_ctx.repo, commit=graph_ctx.commit, repo_root=graph_ctx.repo_root
        ),
    ).graph_metrics()

    compute_graph_metrics(graph_ctx.gateway, cfg)

    row_count = graph_ctx.query_count(
        "analytics.graph_metrics_functions",
        f"repo = '{graph_ctx.repo}' AND commit = '{graph_ctx.commit}'",
    )

    expect_equal(
        row_count,
        EXPECTED_GRAPH_METRICS_ROWS,
        label=f"Expected {EXPECTED_GRAPH_METRICS_ROWS} function metric rows, got {row_count}",
    )
