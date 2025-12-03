"""Integration test ensuring span alignment across graph builders."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.storage.gateway import StorageGateway
from tests._helpers.graph_env import (
    build_span_graph_components,
    collect_span_snapshot,
    create_span_test_env,
    generate_span_coverage,
)

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"


def test_span_alignment_across_components(tmp_path: Path, fresh_gateway: StorageGateway) -> None:
    """
    Ensure call graph, CFG/DFG, and test coverage edges agree on function GOIDs.

    Raises
    ------
    AssertionError
        If any component produces mismatched GOIDs for the same function spans.
    """
    env = create_span_test_env(tmp_path, fresh_gateway)
    build_span_graph_components(env)
    coverage_artifact = generate_span_coverage(env.repo_root)
    compute_test_coverage_edges(
        env.gateway,
        env.builder.test_coverage(coverage_file=coverage_artifact.coverage_file),
    )

    snapshot = collect_span_snapshot(env.gateway.con)

    expected = {env.expected_goid}
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
