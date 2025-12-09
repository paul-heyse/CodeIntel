"""Hotspots metric computation tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.compute.hotspots.metrics import build_hotspots
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import HotspotsStepConfig
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_length,
    expect_true,
)
from tests._helpers.gateway import GatewayFactory


def test_build_hotspots_inserts_scores() -> None:
    """Build hotspots table from core.ast_metrics with git scan disabled."""
    gateway = GatewayFactory().with_snapshot(repo="demo", commit="abc123").open()
    con = gateway.con
    con.execute("DELETE FROM core.ast_metrics")
    con.execute("DELETE FROM analytics.hotspots")

    generated_at = datetime.now(tz=UTC)
    con.execute(
        "INSERT INTO core.ast_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("src/main.py", 10, 2, 0, 1.0, 2, 3.0, generated_at),
    )

    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=Path.cwd())
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)
    build_hotspots(gateway, cfg)

    rows = con.execute(
        "SELECT rel_path, commit_count, author_count, complexity, score FROM analytics.hotspots"
    ).fetchall()
    expect_length(rows, 1)
    rel_path, commit_count, author_count, complexity, score = rows[0]
    expect_equal((rel_path, commit_count, author_count, complexity), ("src/main.py", 0, 0, 3.0))
    expect_true(score > 0.0)


def test_build_hotspots_logs_git_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Git failures warn but do not abort hotspot computation."""

    class _FailingResult:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = "ok"
            self.stderr = "fatal: not a git repo"

    class _FailingRunner:
        def __init__(self) -> None:
            self._result = _FailingResult()

        def run(self, tool: str, args: list[str], cwd: Path) -> _FailingResult:  # noqa: ARG002
            return self._result

    gateway = GatewayFactory().with_snapshot(repo="demo", commit="abc123").open()
    con = gateway.con
    con.execute("DELETE FROM core.ast_metrics")
    con.execute("DELETE FROM analytics.hotspots")

    generated_at = datetime.now(tz=UTC)
    con.execute(
        "INSERT INTO core.ast_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("src/lib.py", 5, 1, 0, 1.0, 1, 2.0, generated_at),
    )

    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=Path.cwd())
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=5)

    caplog.set_level("WARNING", logger="codeintel.analytics.compute.hotspots.metrics")
    build_hotspots(gateway, cfg, runner=_FailingRunner())

    rows = con.execute(
        "SELECT rel_path, commit_count, author_count, complexity FROM analytics.hotspots"
    ).fetchall()
    expect_length(rows, 1)
    rel_path, commit_count, author_count, complexity = rows[0]
    expect_equal((rel_path, commit_count, author_count, complexity), ("src/lib.py", 0, 0, 2.0))
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="git log exited with code 2",
    )
