"""Tests for coverage aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.coverage import compute_coverage_functions
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
)
from tests._helpers.coverage import (
    CoverageLineSeedData,
    GoidSeedData,
    seed_coverage_line,
    seed_goid,
)
from tests._helpers.gateway import GatewayFactory


def _build_gateway() -> GatewayFactory:
    return GatewayFactory().without_validation().without_views()


@pytest.fixture
def gateway() -> Iterator[StorageGateway]:
    """Provide a writable in-memory gateway for analytics scenarios.

    Yields
    ------
    StorageGateway
        Gateway with schemas applied and validation disabled.
    """
    gw = _build_gateway().open()
    try:
        yield gw
    finally:
        gw.close()


def test_compute_coverage_functions_populates_metrics(gateway: StorageGateway) -> None:
    """Aggregate executable and covered lines into coverage_functions."""
    repo_root = Path.cwd()
    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo="demo/repo", commit="abc123", repo_root=repo_root),
    ).coverage_analytics()
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=repo_root)
    con = gateway.con
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            urn="urn:func",
            rel_path="pkg/mod.py",
            kind="function",
            qualname="pkg.mod.fn",
            goid_h128=1,
            start_line=1,
            end_line=3,
        ),
    )
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            urn="urn:method",
            rel_path="pkg/mod.py",
            kind="method",
            qualname="pkg.mod.method",
            goid_h128=2,
            start_line=10,
            end_line=12,
        ),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 1, is_executable=True, is_covered=True),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 2, is_executable=True, is_covered=False),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 3, is_executable=False, is_covered=False),
    )

    compute_coverage_functions(gateway, cfg)

    rows = con.execute(
        """
        SELECT function_goid_h128, executable_lines, covered_lines,
               coverage_ratio, tested, untested_reason
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        ORDER BY function_goid_h128
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()

    expect_equal(
        rows,
        [
            (1, 2, 1, pytest.approx(0.5), True, ""),
            (2, 0, 0, None, False, "no_executable_code"),
        ],
    )


def test_compute_coverage_functions_idempotent_for_snapshot(gateway: StorageGateway) -> None:
    """Re-running coverage aggregation replaces prior rows for the snapshot."""
    repo_root = Path.cwd()
    cfg = ConfigBuilder.from_snapshot(
        snapshot=SnapshotInit(repo="demo/repo", commit="abc123", repo_root=repo_root),
    ).coverage_analytics()
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=repo_root)
    con = gateway.con
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            urn="urn:func",
            rel_path="pkg/second.py",
            kind="function",
            qualname="pkg.second.fn",
            goid_h128=3,
            start_line=1,
            end_line=1,
        ),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/second.py", 1, is_executable=True, is_covered=False),
    )

    compute_coverage_functions(gateway, cfg)
    con.execute(
        """
        UPDATE analytics.coverage_lines
        SET is_covered = TRUE
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    )
    compute_coverage_functions(gateway, cfg)

    result = con.execute(
        """
        SELECT executable_lines, covered_lines, tested
        FROM analytics.coverage_functions
        WHERE function_goid_h128 = 3
        """
    ).fetchone()

    expect_equal(result, (1, 1, True))
