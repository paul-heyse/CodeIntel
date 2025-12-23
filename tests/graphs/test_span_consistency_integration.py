"""Integration test ensuring span alignment across graph builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.analytics.testing.coverage.edges import (
    TestCoverageOptions,
    build_test_coverage_edges_rows,
)
from tests._helpers.orchestration import build_span_graph_components, collect_span_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.configs import SpanTestEnv

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"
TEST_COVERAGE_EDGES_TABLE_KEY: Final = "analytics.test_coverage_edges"


def test_span_alignment_across_components(
    span_env: SpanTestEnv, span_coverage_artifact: Path
) -> None:
    """
    Ensure call graph, CFG/DFG, and test coverage edges agree on function GOIDs.

    Raises
    ------
    AssertionError
        If any component produces mismatched GOIDs for the same function spans.
    """
    build_span_graph_components(span_env)
    rows = build_test_coverage_edges_rows(
        span_env.gateway,
        span_env.builder.snapshot,
        options=TestCoverageOptions(coverage_file=span_coverage_artifact),
    )
    if rows:
        backend = span_env.gateway.policy
        backend.delete_for_snapshot(
            TEST_COVERAGE_EDGES_TABLE_KEY,
            repo=span_env.builder.snapshot.repo,
            commit=span_env.builder.snapshot.commit,
        )
        backend.bulk_insert_mappings(TEST_COVERAGE_EDGES_TABLE_KEY, rows)

    snapshot = collect_span_snapshot(span_env.gateway.con)

    goid_row = span_env.gateway.con.execute(
        """
        SELECT goid_h128
        FROM core.goids
        WHERE repo = ? AND commit = ? AND qualname = 'pkg.b.caller'
        LIMIT 1
        """,
        [span_env.builder.snapshot.repo, span_env.builder.snapshot.commit],
    ).fetchone()
    if goid_row is None:
        message = "Expected GOID for pkg.b.caller to be present"
        raise AssertionError(message)
    expected = {int(goid_row[0])}
    if snapshot.cfg_goids != expected:
        message = f"CFG goids mismatch: expected {expected}, got {snapshot.cfg_goids}"
        raise AssertionError(message)
    if snapshot.callgraph_goids != expected:
        message = f"Call graph goids mismatch: expected {expected}, got {snapshot.callgraph_goids}"
        raise AssertionError(message)
    if not expected.issubset(snapshot.coverage_goids):
        missing = expected - snapshot.coverage_goids
        message = (
            "Coverage goids mismatch: expected superset "
            f"{expected}, missing {missing}, got {snapshot.coverage_goids}"
        )
        raise AssertionError(message)
    if snapshot.symbol_use_paths != {"pkg/b.py"}:
        message = f"Symbol use mapping mismatch: expected pkg/b.py, got {snapshot.symbol_use_paths}"
        raise AssertionError(message)
