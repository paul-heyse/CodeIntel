"""Smoke tests for graph metrics aggregation."""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs import compute_graph_metrics
from codeintel.config import ConfigBuilder, GraphMetricsStepConfig
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ImportGraphEdgeRow,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_import_graph_edges,
)
from tests._helpers.gateway import open_ingestion_gateway


def test_compute_graph_metrics_with_small_graph() -> None:
    """Verify graph metrics populate basic call/import rows."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    con = gateway.con
    con.execute(
        "DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    )
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
    from pathlib import Path
    cfg = ConfigBuilder.from_snapshot(repo="demo/repo", commit="deadbeef", repo_root=Path(".")).graph_metrics()
    compute_graph_metrics(gateway, cfg)
    rows = con.execute(
        "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    ).fetchone()
    expected_rows = 2
    if rows is None or rows[0] != expected_rows:
        pytest.fail(f"Expected {expected_rows} function metric rows, got {rows}")
    gateway.close()
