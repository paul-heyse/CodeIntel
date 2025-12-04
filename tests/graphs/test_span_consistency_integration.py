"""Integration test ensuring span alignment across graph builders."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from codeintel.analytics.tests import compute_test_coverage_edges
from tests._helpers.configs import SpanTestEnv
from tests._helpers.orchestration import build_span_graph_components, collect_span_snapshot

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"


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
    compute_test_coverage_edges(
        span_env.gateway,
        span_env.builder.test_coverage(coverage_file=span_coverage_artifact),
    )

    snapshot = collect_span_snapshot(span_env.gateway.con)

    expected = {span_env.expected_goid}
    if snapshot.cfg_goids != expected:
        message = f"CFG goids mismatch: expected {expected}, got {snapshot.cfg_goids}"
        raise AssertionError(message)
    if snapshot.callgraph_goids != expected:
        message = f"Call graph goids mismatch: expected {expected}, got {snapshot.callgraph_goids}"
        raise AssertionError(message)
    if snapshot.coverage_goids != expected:
        message = f"Coverage goids mismatch: expected {expected}, got {snapshot.coverage_goids}"
        raise AssertionError(message)
    if snapshot.symbol_use_paths != {"pkg/b.py"}:
        message = f"Symbol use mapping mismatch: expected pkg/b.py, got {snapshot.symbol_use_paths}"
        raise AssertionError(message)
