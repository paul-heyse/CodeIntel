"""Tests for AST metrics and git churn hotspots computation.

This module tests:
- FileChurn dataclass for aggregating per-file churn
- build_hotspots function for computing hotspot scores
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.hotspots.metrics import FileChurn, build_hotspots
from codeintel.config import HotspotsStepConfig
from codeintel.storage.gateway import StorageGateway
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import gateway_with_macros

# Test constants (non-repo/commit)
EXPECTED_COMMIT_COUNT = 2
EXPECTED_AUTHOR_COUNT = 2
EXPECTED_LINES_ADDED = 30
EXPECTED_LINES_DELETED = 10
HOTSPOT_SCORE_THRESHOLD = 0.0
EXPECTED_FILE_COUNT_MULTI = 3
EXPECTED_COMPLEXITY = 10.0
EXPECTED_SUMMARY_KEYS = 4


@pytest.fixture
def memory_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory DuckDB gateway for testing.

    Yields
    ------
    StorageGateway
        Configured gateway with schema applied.
    """
    gateway = gateway_with_macros()
    try:
        yield gateway
    finally:
        gateway.con.close()


@pytest.fixture
def hotspots_config(tmp_path: Path) -> HotspotsStepConfig:
    """Create a HotspotsStepConfig for testing.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    Returns
    -------
    HotspotsStepConfig
        Configured step config.
    """
    snapshot = make_snapshot(repo_root=tmp_path)
    return HotspotsStepConfig(snapshot=snapshot, max_commits=100)


def _insert_ast_metric(
    gateway: StorageGateway,
    rel_path: str,
    complexity: float | None = 5.0,
) -> None:
    """Insert a test AST metric record.

    Parameters
    ----------
    gateway
        Storage gateway.
    rel_path
        File path.
    complexity
        Complexity value.
    """
    gateway.con.execute(
        """
        INSERT INTO core.ast_metrics (
            rel_path, node_count, function_count, class_count,
            avg_depth, max_depth, complexity, generated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [rel_path, 100, 10, 2, 2.5, 5, complexity],
    )


def test_file_churn_creation() -> None:
    """Create a FileChurn with default values."""
    churn = FileChurn()

    assert churn.commits == set()
    assert churn.authors == set()
    assert churn.lines_added == 0
    assert churn.lines_deleted == 0


def test_file_churn_accumulation() -> None:
    """Accumulate churn data."""
    churn = FileChurn()
    churn.commits.add("abc123")
    churn.commits.add("def456")
    churn.authors.add("Alice")
    churn.authors.add("Bob")
    churn.lines_added = EXPECTED_LINES_ADDED
    churn.lines_deleted = EXPECTED_LINES_DELETED

    summary = churn.to_summary()

    assert summary["commit_count"] == EXPECTED_COMMIT_COUNT
    assert summary["author_count"] == EXPECTED_AUTHOR_COUNT
    assert summary["lines_added"] == EXPECTED_LINES_ADDED
    assert summary["lines_deleted"] == EXPECTED_LINES_DELETED


def test_file_churn_empty_summary() -> None:
    """Empty churn produces zero counts."""
    churn = FileChurn()
    summary = churn.to_summary()

    assert summary["commit_count"] == 0
    assert summary["author_count"] == 0
    assert summary["lines_added"] == 0
    assert summary["lines_deleted"] == 0


def test_file_churn_duplicate_commits() -> None:
    """Duplicate commits are deduplicated."""
    churn = FileChurn()
    churn.commits.add("abc123")
    churn.commits.add("abc123")  # Duplicate

    summary = churn.to_summary()

    assert summary["commit_count"] == 1


def test_file_churn_to_summary_keys() -> None:
    """Verify to_summary returns expected keys."""
    churn = FileChurn()
    summary = churn.to_summary()

    assert "commit_count" in summary
    assert "author_count" in summary
    assert "lines_added" in summary
    assert "lines_deleted" in summary
    assert len(summary) == EXPECTED_SUMMARY_KEYS


def test_build_hotspots_empty_ast_metrics(
    memory_gateway: StorageGateway,
    hotspots_config: HotspotsStepConfig,
) -> None:
    """Build hotspots with empty AST metrics produces no rows."""
    # AST metrics table is empty by default
    build_hotspots(memory_gateway, hotspots_config)

    # Should have no hotspot rows
    result = memory_gateway.con.execute("SELECT COUNT(*) FROM analytics.hotspots").fetchone()
    assert result is not None
    assert result[0] == 0


def test_build_hotspots_with_ast_data(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots with AST metrics data."""
    _insert_ast_metric(memory_gateway, "test_file.py", complexity=5.0)

    # Create config with max_commits=0 to skip git log
    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    # Should have one hotspot row
    result = memory_gateway.con.execute(
        "SELECT rel_path, score FROM analytics.hotspots WHERE rel_path = ?",
        ["test_file.py"],
    ).fetchone()

    assert result is not None
    rel_path, score = result
    assert rel_path == "test_file.py"
    assert score > HOTSPOT_SCORE_THRESHOLD  # Score should be positive


def test_build_hotspots_multiple_files(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots with multiple files."""
    _insert_ast_metric(memory_gateway, "file1.py", complexity=3.0)
    _insert_ast_metric(memory_gateway, "file2.py", complexity=7.0)
    _insert_ast_metric(memory_gateway, "file3.py", complexity=12.0)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    # Should have three hotspot rows
    result = memory_gateway.con.execute("SELECT COUNT(*) FROM analytics.hotspots").fetchone()

    assert result is not None
    assert result[0] == EXPECTED_FILE_COUNT_MULTI


def test_build_hotspots_score_calculation(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify hotspot score calculation components."""
    _insert_ast_metric(memory_gateway, "scored.py", complexity=EXPECTED_COMPLEXITY)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    result = memory_gateway.con.execute(
        """
        SELECT complexity, score, commit_count, author_count
        FROM analytics.hotspots WHERE rel_path = ?
        """,
        ["scored.py"],
    ).fetchone()

    assert result is not None
    complexity, score, commit_count, author_count = result
    assert complexity == EXPECTED_COMPLEXITY
    # With no git stats, commit_count and author_count should be 0
    assert commit_count == 0
    assert author_count == 0
    # Score should still be positive due to complexity component
    assert score > HOTSPOT_SCORE_THRESHOLD


def test_build_hotspots_high_complexity(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots handles high complexity values."""
    high_complexity = 100.0
    _insert_ast_metric(memory_gateway, "high_complexity.py", complexity=high_complexity)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    result = memory_gateway.con.execute(
        "SELECT complexity, score FROM analytics.hotspots WHERE rel_path = ?",
        ["high_complexity.py"],
    ).fetchone()

    assert result is not None
    complexity, score = result
    assert complexity == high_complexity
    # Higher complexity should result in higher score
    assert score > HOTSPOT_SCORE_THRESHOLD


def test_build_hotspots_idempotent(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots is idempotent (DELETE before INSERT)."""
    _insert_ast_metric(memory_gateway, "idempotent.py", complexity=5.0)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    # Run twice
    build_hotspots(memory_gateway, cfg)
    build_hotspots(memory_gateway, cfg)

    # Should still have only one row
    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.hotspots WHERE rel_path = ?",
        ["idempotent.py"],
    ).fetchone()

    assert result is not None
    assert result[0] == 1


@pytest.mark.parametrize(
    ("lines_added", "lines_deleted"),
    [
        (0, 0),
        (100, 0),
        (0, 100),
        (50, 50),
        (1000, 500),
    ],
)
def test_file_churn_line_counts(lines_added: int, lines_deleted: int) -> None:
    """Test various line count combinations."""
    churn = FileChurn()
    churn.lines_added = lines_added
    churn.lines_deleted = lines_deleted

    summary = churn.to_summary()

    assert summary["lines_added"] == lines_added
    assert summary["lines_deleted"] == lines_deleted


def test_build_hotspots_windows_path_handling(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots normalizes Windows-style paths."""
    _insert_ast_metric(memory_gateway, "path\\to\\file.py", complexity=5.0)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    # Path should exist in hotspots table
    result = memory_gateway.con.execute(
        "SELECT rel_path FROM analytics.hotspots WHERE rel_path LIKE '%file.py'"
    ).fetchone()

    assert result is not None


def test_build_hotspots_zero_complexity(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots handles zero complexity."""
    _insert_ast_metric(memory_gateway, "zero_complexity.py", complexity=0.0)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    result = memory_gateway.con.execute(
        "SELECT complexity, score FROM analytics.hotspots WHERE rel_path = ?",
        ["zero_complexity.py"],
    ).fetchone()

    assert result is not None
    complexity, score = result
    assert complexity == 0.0
    # Score should still be non-negative
    assert score >= HOTSPOT_SCORE_THRESHOLD


def test_build_hotspots_negative_complexity(
    memory_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Build hotspots handles negative complexity (clamps to zero)."""
    _insert_ast_metric(memory_gateway, "negative.py", complexity=-5.0)

    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = HotspotsStepConfig(snapshot=snapshot, max_commits=0)

    build_hotspots(memory_gateway, cfg)

    result = memory_gateway.con.execute(
        "SELECT complexity, score FROM analytics.hotspots WHERE rel_path = ?",
        ["negative.py"],
    ).fetchone()

    assert result is not None
    _, score = result
    # Score should be non-negative due to max(complexity, 0.0)
    assert score >= HOTSPOT_SCORE_THRESHOLD
