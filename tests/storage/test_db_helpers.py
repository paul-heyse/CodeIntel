"""Tests for db_helpers module."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.db_helpers import (
    row_counts_for_tables,
    safe_row_counts,
)
from codeintel.storage.gateway import StorageGateway


def test_row_counts_for_tables_returns_dict(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_counts_for_tables returns dict with counts."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    now = datetime.now(tz=UTC)
    now_str = now.isoformat()

    fresh_gateway.core.insert_modules([("test_mod", "test.py", repo, commit)])
    fresh_gateway.core.insert_repo_map([(repo, commit, "{}", "{}", now_str)])

    tables = ["core.modules", "core.repo_map"]
    result = row_counts_for_tables(con, repo=repo, commit=commit, tables=tables)

    assert result is not None
    assert isinstance(result, dict)
    assert "core.modules" in result
    assert "core.repo_map" in result


def test_row_counts_for_tables_filters_by_repo_commit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_counts_for_tables filters by repo and commit."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"
    other_repo = "other/repo"
    other_commit = "def456"

    fresh_gateway.core.insert_modules([("test_mod", "test.py", repo, commit)])
    fresh_gateway.core.insert_modules([("other_mod", "other.py", other_repo, other_commit)])

    tables = ["core.modules"]
    result = row_counts_for_tables(con, repo=repo, commit=commit, tables=tables)

    assert result is not None
    assert result.get("core.modules") == 1


def test_row_counts_for_tables_returns_none_on_missing_table(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_counts_for_tables returns None when table doesn't exist."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    tables = ["nonexistent.table"]
    result = row_counts_for_tables(con, repo=repo, commit=commit, tables=tables)

    assert result is None


def test_row_counts_for_tables_handles_special_chars_in_repo(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_counts_for_tables handles special characters in repo."""
    con = fresh_gateway.con
    repo = "test/repo's-name"
    commit = "abc123"

    fresh_gateway.core.insert_modules([("test_mod", "test.py", repo, commit)])

    tables = ["core.modules"]
    result = row_counts_for_tables(con, repo=repo, commit=commit, tables=tables)

    assert result is not None
    assert result.get("core.modules") == 1


def test_row_counts_for_tables_handles_empty_tables(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_counts_for_tables handles empty tables correctly."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    tables = ["core.modules"]
    result = row_counts_for_tables(con, repo=repo, commit=commit, tables=tables)

    assert result is not None
    assert result.get("core.modules") == 0


def test_safe_row_counts_tolerates_none_connection() -> None:
    """Verify safe_row_counts returns None when connection is None."""
    result = safe_row_counts(None, repo="test/repo", commit="abc123", tables=["core.modules"])

    assert result is None


def test_safe_row_counts_returns_counts_with_valid_connection(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_row_counts returns counts with valid connection."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    fresh_gateway.core.insert_modules([("test_mod", "test.py", repo, commit)])

    result = safe_row_counts(con, repo=repo, commit=commit, tables=["core.modules"])

    assert result is not None
    assert "core.modules" in result


def test_safe_row_counts_accepts_iterable_tables(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_row_counts accepts any iterable for tables."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    fresh_gateway.core.insert_modules([("test_mod", "test.py", repo, commit)])

    tables_set = {"core.modules", "core.repo_map"}
    result = safe_row_counts(con, repo=repo, commit=commit, tables=tables_set)

    assert result is not None
    expected_table_count = 2
    assert len(result) == expected_table_count


def test_safe_row_counts_returns_none_on_table_error(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_row_counts returns None when table operation fails."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    result = safe_row_counts(con, repo=repo, commit=commit, tables=["nonexistent.table"])

    assert result is None
