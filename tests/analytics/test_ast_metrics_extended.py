"""Extended tests for AST metrics computation.

Test the AST-derived metrics and git churn hotspot computation.

This module tests internal functions that parse git log output. These tests
intentionally import private functions as they exercise low-level parsing
behavior that may change with implementation details.
"""

from __future__ import annotations

# We test internal functions to verify git log parsing behavior.
# ruff: noqa: PLC2701
from codeintel.analytics.ast_metrics import FileChurn, _parse_git_log_lines

# =============================================================================
# Test Constants
# =============================================================================

EXPECTED_COMMITS_1 = 1
EXPECTED_COMMITS_2 = 2
EXPECTED_AUTHORS_1 = 1
EXPECTED_AUTHORS_2 = 2
EXPECTED_AUTHORS_3 = 3
EXPECTED_LINES_ADDED_10 = 10
EXPECTED_LINES_ADDED_15 = 15
EXPECTED_LINES_ADDED_100 = 100
EXPECTED_LINES_DELETED_0 = 0
EXPECTED_LINES_DELETED_3 = 3
EXPECTED_LINES_DELETED_5 = 5
EXPECTED_LINES_DELETED_50 = 50
EXPECTED_FILES_2 = 2


# =============================================================================
# FileChurn Tests
# =============================================================================


def test_file_churn_default_values() -> None:
    """Verify FileChurn has sensible defaults."""
    churn = FileChurn()
    assert not churn.commits
    assert not churn.authors
    assert churn.lines_added == EXPECTED_LINES_DELETED_0
    assert churn.lines_deleted == EXPECTED_LINES_DELETED_0


def test_file_churn_to_summary() -> None:
    """Convert FileChurn to summary dict."""
    churn = FileChurn()
    churn.commits.add("abc123")
    churn.commits.add("def456")
    churn.authors.add("alice")
    churn.lines_added = EXPECTED_LINES_ADDED_100
    churn.lines_deleted = EXPECTED_LINES_DELETED_50

    summary = churn.to_summary()
    assert summary["commit_count"] == EXPECTED_COMMITS_2
    assert summary["author_count"] == EXPECTED_AUTHORS_1
    assert summary["lines_added"] == EXPECTED_LINES_ADDED_100
    assert summary["lines_deleted"] == EXPECTED_LINES_DELETED_50


def test_file_churn_multiple_authors() -> None:
    """Track multiple distinct authors."""
    churn = FileChurn()
    churn.authors.add("alice")
    churn.authors.add("bob")
    churn.authors.add("charlie")
    summary = churn.to_summary()
    assert summary["author_count"] == EXPECTED_AUTHORS_3


# =============================================================================
# _parse_git_log_lines Tests
# =============================================================================


def test_parse_git_log_empty_lines() -> None:
    """Empty input returns empty dict."""
    result = _parse_git_log_lines([])
    assert not result


def test_parse_git_log_simple_commit() -> None:
    """Parse a single commit with one file."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t5\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    assert "src/module.py" in result
    stats = result["src/module.py"]
    assert stats["commit_count"] == EXPECTED_COMMITS_1
    assert stats["author_count"] == EXPECTED_AUTHORS_1
    assert stats["lines_added"] == EXPECTED_LINES_ADDED_10
    assert stats["lines_deleted"] == EXPECTED_LINES_DELETED_5


def test_parse_git_log_multiple_commits_same_file() -> None:
    """Aggregate stats across multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tBob",
        "5\t3\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    stats = result["src/module.py"]
    assert stats["commit_count"] == EXPECTED_COMMITS_2
    assert stats["author_count"] == EXPECTED_AUTHORS_2
    assert stats["lines_added"] == EXPECTED_LINES_ADDED_15
    assert stats["lines_deleted"] == EXPECTED_LINES_DELETED_3


def test_parse_git_log_multiple_files() -> None:
    """Parse commits touching multiple files."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/a.py",
        "20\t5\tsrc/b.py",
    ]
    result = _parse_git_log_lines(lines)
    assert len(result) == EXPECTED_FILES_2
    assert "src/a.py" in result
    assert "src/b.py" in result


def test_parse_git_log_empty_line_handling() -> None:
    """Skip empty lines in input."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "",
        "10\t0\tsrc/module.py",
        "",
    ]
    result = _parse_git_log_lines(lines)
    assert "src/module.py" in result


def test_parse_git_log_binary_file_numstat() -> None:
    """Handle binary files (- in numstat)."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "-\t-\timage.png",  # Binary files show - - path
        "10\t0\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    # Binary file should have 0 lines (- is not a digit)
    assert "image.png" in result
    stats = result["image.png"]
    assert stats["lines_added"] == EXPECTED_LINES_DELETED_0


def test_parse_git_log_normalize_path_separators() -> None:
    """Backslashes are normalized to forward slashes."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc\\windows\\module.py",
    ]
    result = _parse_git_log_lines(lines)
    assert "src/windows/module.py" in result


def test_parse_git_log_skip_lines_before_commit() -> None:
    """Skip numstat lines before any COMMIT header."""
    lines = [
        "10\t0\torphan.py",  # No commit yet
        "COMMIT\tabc123\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    assert "orphan.py" not in result
    assert "src/module.py" in result


def test_parse_git_log_malformed_numstat_line() -> None:
    """Skip malformed numstat lines."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "not\ta\tvalid\tnumstat",  # 4 fields, not 3
        "10\t0\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    assert len(result) == EXPECTED_COMMITS_1


def test_parse_git_log_same_author_multiple_commits() -> None:
    """Same author counted once even with multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = _parse_git_log_lines(lines)
    stats = result["src/module.py"]
    assert stats["commit_count"] == EXPECTED_COMMITS_2
    assert stats["author_count"] == EXPECTED_AUTHORS_1
