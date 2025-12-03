"""Smoke test for graph stats population."""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs import compute_graph_stats
from tests._helpers.context import TestContext
from tests._helpers.seeds import GRAPH_PACK


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
