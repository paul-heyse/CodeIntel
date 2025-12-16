"""Tests for db_helpers module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.storage.validation import (
    count_rows_for_tables,
    safe_count_rows,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
)
from tests._helpers.builders import ModuleRow, RepoMapRow, insert_rows

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_count_rows_for_tables_returns_dict(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify count_rows_for_tables returns dict with counts."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    now = datetime.now(tz=UTC)

    insert_rows(
        fresh_gateway,
        [
            ModuleRow(module="test_mod", path="test.py", repo=repo, commit=commit),
            RepoMapRow(repo=repo, commit=commit, modules={}, overlays={}, generated_at=now),
        ],
    )

    tables = ["core.modules", "core.repo_map"]
    result = count_rows_for_tables(con, repo=repo, commit=commit, tables=tables)

    expect_is_not_none(result)
    expect_is_instance(result, dict)
    expect_in("core.modules", result or {})
    expect_in("core.repo_map", result or {})


def test_count_rows_for_tables_filters_by_repo_commit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify count_rows_for_tables filters by repo and commit."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"
    other_repo = "other/repo"
    other_commit = "def456"

    insert_rows(
        fresh_gateway,
        [
            ModuleRow(module="test_mod", path="test.py", repo=repo, commit=commit),
            ModuleRow(module="other_mod", path="other.py", repo=other_repo, commit=other_commit),
        ],
    )

    tables = ["core.modules"]
    result = count_rows_for_tables(con, repo=repo, commit=commit, tables=tables)

    expect_is_not_none(result)
    if result is None:
        return
    expect_equal(result.get("core.modules"), 1)


def test_count_rows_for_tables_returns_none_on_missing_table(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify count_rows_for_tables returns None when table doesn't exist."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    tables = ["nonexistent.table"]
    result = count_rows_for_tables(con, repo=repo, commit=commit, tables=tables)

    expect_is_none(result)


def test_count_rows_for_tables_handles_special_chars_in_repo(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify count_rows_for_tables handles special characters in repo."""
    con = fresh_gateway.con
    repo = "test/repo's-name"
    commit = "abc123"

    insert_rows(
        fresh_gateway,
        [ModuleRow(module="test_mod", path="test.py", repo=repo, commit=commit)],
    )

    tables = ["core.modules"]
    result = count_rows_for_tables(con, repo=repo, commit=commit, tables=tables)

    expect_is_not_none(result)
    if result is None:
        return
    expect_equal(result.get("core.modules"), 1)


def test_count_rows_for_tables_handles_empty_tables(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify count_rows_for_tables handles empty tables correctly."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    tables = ["core.modules"]
    result = count_rows_for_tables(con, repo=repo, commit=commit, tables=tables)

    expect_is_not_none(result)
    if result is None:
        return
    expect_equal(result.get("core.modules"), 0)


def test_safe_count_rows_tolerates_none_connection() -> None:
    """Verify safe_count_rows returns None when connection is None."""
    result = safe_count_rows(None, repo="test/repo", commit="abc123", tables=["core.modules"])

    expect_is_none(result)


def test_safe_count_rows_returns_counts_with_valid_connection(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_count_rows returns counts with valid connection."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    insert_rows(
        fresh_gateway,
        [ModuleRow(module="test_mod", path="test.py", repo=repo, commit=commit)],
    )

    result = safe_count_rows(con, repo=repo, commit=commit, tables=["core.modules"])

    expect_is_not_none(result)
    expect_in("core.modules", result or {})


def test_safe_count_rows_accepts_iterable_tables(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_count_rows accepts any iterable for tables."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    insert_rows(
        fresh_gateway,
        [ModuleRow(module="test_mod", path="test.py", repo=repo, commit=commit)],
    )

    tables_set = {"core.modules", "core.repo_map"}
    result = safe_count_rows(con, repo=repo, commit=commit, tables=tables_set)

    expect_is_not_none(result)
    expect_length(result or {}, 2)


def test_safe_count_rows_returns_none_on_table_error(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify safe_count_rows returns None when table operation fails."""
    con = fresh_gateway.con
    repo = "test/repo"
    commit = "abc123"

    result = safe_count_rows(con, repo=repo, commit=commit, tables=["nonexistent.table"])

    expect_is_none(result)
