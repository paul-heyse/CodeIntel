"""Smoke test for graph stats population."""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs import compute_graph_stats
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ImportGraphEdgeRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_import_graph_edges,
)
from tests._helpers.gateway import open_ingestion_gateway


def test_graph_stats_records_counts_for_basic_graphs() -> None:
    """Ensure graph_stats has entries after minimal graph ingestion."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    con = gateway.con
    insert_call_graph_nodes(
        gateway,
        [
            CallGraphNodeRow(1, "python", "function", 0, is_public=True, rel_path="pkg/a.py"),
            CallGraphNodeRow(2, "python", "function", 0, is_public=True, rel_path="pkg/b.py"),
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                repo="demo/repo",
                commit="deadbeef",
                caller_goid_h128=1,
                callee_goid_h128=2,
                callsite_path="pkg/a.py",
                callsite_line=1,
                callsite_col=0,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
            )
        ],
    )
    insert_import_graph_edges(
        gateway,
        [
            ImportGraphEdgeRow(
                repo="demo/repo",
                commit="deadbeef",
                src_module="pkg.a",
                dst_module="pkg.b",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            )
        ],
    )

    compute_graph_stats(gateway, repo="demo/repo", commit="deadbeef")
    rows = con.execute(
        "SELECT graph_name, node_count FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    ).fetchall()
    if not rows:
        pytest.fail("Expected graph_stats to contain rows for ingested graphs")
    gateway.close()
