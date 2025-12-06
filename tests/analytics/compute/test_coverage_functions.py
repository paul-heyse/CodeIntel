"""Tests for coverage aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.coverage import compute_coverage_functions
from codeintel.config import ConfigBuilder
from codeintel.storage.gateway import StorageGateway
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
    cfg = ConfigBuilder.from_snapshot(
        repo="demo/repo", commit="abc123", repo_root=Path.cwd()
    ).coverage_analytics()
    con = gateway.con
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, created_at
        ) VALUES
            (1, 'urn:func', ?, ?, 'pkg/mod.py', 'python', 'function',
             'pkg.mod.fn', 1, 3, NOW()),
            (2, 'urn:method', ?, ?, 'pkg/mod.py', 'python', 'method',
             'pkg.mod.method', 10, 12, NOW())
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit],
    )
    con.execute(
        """
        INSERT INTO analytics.coverage_lines (
            repo, commit, rel_path, line, is_executable, is_covered, hits,
            context_count, created_at
        ) VALUES
            (?, ?, 'pkg/mod.py', 1, TRUE, TRUE, 1, 0, NOW()),
            (?, ?, 'pkg/mod.py', 2, TRUE, FALSE, 0, 0, NOW()),
            (?, ?, 'pkg/mod.py', 3, FALSE, FALSE, 0, 0, NOW())
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit, cfg.repo, cfg.commit],
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

    assert rows == [
        (1, 2, 1, pytest.approx(0.5), True, ""),
        (2, 0, 0, None, False, "no_executable_code"),
    ]


def test_compute_coverage_functions_idempotent_for_snapshot(gateway: StorageGateway) -> None:
    """Re-running coverage aggregation replaces prior rows for the snapshot."""
    cfg = ConfigBuilder.from_snapshot(
        repo="demo/repo", commit="abc123", repo_root=Path.cwd()
    ).coverage_analytics()
    con = gateway.con
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, created_at
        ) VALUES
            (3, 'urn:func', ?, ?, 'pkg/second.py', 'python', 'function',
             'pkg.second.fn', 1, 1, NOW())
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        INSERT INTO analytics.coverage_lines (
            repo, commit, rel_path, line, is_executable, is_covered, hits,
            context_count, created_at
        ) VALUES (?, ?, 'pkg/second.py', 1, TRUE, FALSE, 0, 0, NOW())
        """,
        [cfg.repo, cfg.commit],
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

    assert result == (1, 1, True)
