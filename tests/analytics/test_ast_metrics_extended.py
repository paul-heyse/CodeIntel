"""Extended tests for AST metrics computation.

Test the AST-derived metrics and git churn hotspot computation.

This module tests internal functions that parse git log output. These tests
intentionally import private functions as they exercise low-level parsing
behavior that may change with implementation details.
"""

from __future__ import annotations

# We test git log parsing behavior via the public wrapper.
from codeintel.analytics.compute.hotspots.metrics import parse_git_log_lines
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_length,
    expect_not_in,
)

# =============================================================================
# Test Constants
# =============================================================================

EXPECTED_COMMITS_1 = 1
EXPECTED_COMMITS_2 = 2
EXPECTED_AUTHORS_1 = 1
EXPECTED_AUTHORS_2 = 2
EXPECTED_LINES_ADDED_10 = 10
EXPECTED_LINES_ADDED_15 = 15
EXPECTED_LINES_DELETED_0 = 0
EXPECTED_LINES_DELETED_3 = 3
EXPECTED_LINES_DELETED_5 = 5
EXPECTED_FILES_2 = 2


# =============================================================================
# parse_git_log_lines Tests
# =============================================================================


def test_parse_git_log_empty_lines() -> None:
    """Empty input returns empty dict."""
    result = parse_git_log_lines([])
    expect_false(result)


def test_parse_git_log_simple_commit() -> None:
    """Parse a single commit with one file."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t5\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/module.py", result)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], EXPECTED_COMMITS_1)
    expect_equal(stats["author_count"], EXPECTED_AUTHORS_1)
    expect_equal(stats["lines_added"], EXPECTED_LINES_ADDED_10)
    expect_equal(stats["lines_deleted"], EXPECTED_LINES_DELETED_5)


def test_parse_git_log_multiple_commits_same_file() -> None:
    """Aggregate stats across multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tBob",
        "5\t3\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], EXPECTED_COMMITS_2)
    expect_equal(stats["author_count"], EXPECTED_AUTHORS_2)
    expect_equal(stats["lines_added"], EXPECTED_LINES_ADDED_15)
    expect_equal(stats["lines_deleted"], EXPECTED_LINES_DELETED_3)


def test_parse_git_log_multiple_files() -> None:
    """Parse commits touching multiple files."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/a.py",
        "20\t5\tsrc/b.py",
    ]
    result = parse_git_log_lines(lines)
    expect_length(result, EXPECTED_FILES_2)
    expect_in("src/a.py", result)
    expect_in("src/b.py", result)


def test_parse_git_log_empty_line_handling() -> None:
    """Skip empty lines in input."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "",
        "10\t0\tsrc/module.py",
        "",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/module.py", result)


def test_parse_git_log_binary_file_numstat() -> None:
    """Handle binary files (- in numstat)."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "-\t-\timage.png",  # Binary files show - - path
        "10\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    # Binary file should have 0 lines (- is not a digit)
    expect_in("image.png", result)
    stats = result["image.png"]
    expect_equal(stats["lines_added"], EXPECTED_LINES_DELETED_0)


def test_parse_git_log_normalize_path_separators() -> None:
    """Backslashes are normalized to forward slashes."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc\\windows\\module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_in("src/windows/module.py", result)


def test_parse_git_log_skip_lines_before_commit() -> None:
    """Skip numstat lines before any COMMIT header."""
    lines = [
        "10\t0\torphan.py",  # No commit yet
        "COMMIT\tabc123\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_not_in("orphan.py", result)
    expect_in("src/module.py", result)


def test_parse_git_log_malformed_numstat_line() -> None:
    """Skip malformed numstat lines."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "not\ta\tvalid\tnumstat",  # 4 fields, not 3
        "10\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    expect_length(result, EXPECTED_COMMITS_1)


def test_parse_git_log_same_author_multiple_commits() -> None:
    """Same author counted once even with multiple commits."""
    lines = [
        "COMMIT\tabc123\tAlice",
        "10\t0\tsrc/module.py",
        "COMMIT\tdef456\tAlice",
        "5\t0\tsrc/module.py",
    ]
    result = parse_git_log_lines(lines)
    stats = result["src/module.py"]
    expect_equal(stats["commit_count"], EXPECTED_COMMITS_2)
    expect_equal(stats["author_count"], EXPECTED_AUTHORS_1)
