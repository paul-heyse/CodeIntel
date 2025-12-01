"""Smoke tests for snapshot and graph runtime option builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from tests._helpers.config_builders import make_graph_runtime_options, make_snapshot


def test_make_snapshot_respects_overrides(tmp_path: Path) -> None:
    """Snapshot builder should reflect provided repo/commit/root."""
    snapshot = make_snapshot(repo="r1", commit="c1", repo_root=tmp_path)
    if snapshot.repo != "r1" or snapshot.commit != "c1":
        pytest.fail("Snapshot builder did not apply repo/commit overrides")
    if snapshot.repo_root != tmp_path:
        pytest.fail("Snapshot builder did not apply repo_root override")


def test_make_graph_runtime_options_uses_snapshot() -> None:
    """Graph runtime options builder should embed the supplied snapshot."""
    snapshot = make_snapshot(repo="r2", commit="c2", repo_root=Path.cwd())
    options = make_graph_runtime_options(snapshot=snapshot, eager=True)
    if options.snapshot is not snapshot:
        pytest.fail("Runtime options builder did not attach provided snapshot")
    if options.resolved_eager is not True:
        pytest.fail("Runtime options builder should respect overrides")
    if not isinstance(options, GraphRuntimeOptions):
        pytest.fail("Builder must return GraphRuntimeOptions")
