"""Tests for repositories/base.py module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    fetch_all_dicts,
    fetch_one_dict,
    fetch_paginated,
    row_exists,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.rows import module_row

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_fetch_one_dict_returns_mapping(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_one_dict returns single row as dict."""
    con = fresh_gateway.con

    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_row(
                module="test_mod", path="test.py", snapshot=("test/repo", "abc123")
            ).to_tuple()
        ],
    )

    result = fetch_one_dict(
        con,
        "SELECT module, path FROM core.modules WHERE repo = ? AND commit = ?",
        ["test/repo", "abc123"],
    )

    row = expect_is_not_none(result)
    expect_is_instance(row, dict)
    expect_equal(row["module"], "test_mod")
    expect_equal(row["path"], "test.py")


def test_fetch_one_dict_returns_none_when_empty(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_one_dict returns None when no rows match."""
    con = fresh_gateway.con

    result = fetch_one_dict(
        con,
        "SELECT module FROM core.modules WHERE repo = ?",
        ["nonexistent/repo"],
    )

    expect_is_none(result)


def test_fetch_all_dicts_returns_list(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_all_dicts returns all rows as list of dicts."""
    con = fresh_gateway.con

    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_row(module="mod1", path="mod1.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod2", path="mod2.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod3", path="mod3.py", snapshot=("test/repo", "abc123")).to_tuple(),
        ],
    )

    result = fetch_all_dicts(
        con,
        "SELECT module, path FROM core.modules WHERE repo = ? ORDER BY module",
        ["test/repo"],
    )

    expect_is_instance(result, list)
    expected_count = 3
    expect_length(result, expected_count)
    expect_equal(result[0]["module"], "mod1")
    expect_equal(result[1]["module"], "mod2")
    expect_equal(result[2]["module"], "mod3")


def test_fetch_all_dicts_returns_empty_list_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_all_dicts returns empty list when no rows match."""
    con = fresh_gateway.con

    result = fetch_all_dicts(
        con,
        "SELECT module FROM core.modules WHERE repo = ?",
        ["nonexistent/repo"],
    )

    expect_is_instance(result, list)
    expect_length(result, 0)


def test_fetch_paginated_detects_truncation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_paginated detects when more rows exist than limit."""
    con = fresh_gateway.con

    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_row(module="mod1", path="mod1.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod2", path="mod2.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod3", path="mod3.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod4", path="mod4.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod5", path="mod5.py", snapshot=("test/repo", "abc123")).to_tuple(),
        ],
    )

    result = fetch_paginated(
        con,
        "SELECT module FROM core.modules WHERE repo = ? ORDER BY module LIMIT ?",
        ["test/repo"],
        limit=3,
    )

    expect_is_instance(result, PaginatedRows)
    expected_count = 3
    expect_length(result.rows, expected_count)
    expect_equal(result.limit, expected_count)
    expect_true(result.truncated)


def test_fetch_paginated_no_truncation_when_under_limit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_paginated reports no truncation when fewer rows than limit."""
    con = fresh_gateway.con

    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_row(module="mod1", path="mod1.py", snapshot=("test/repo", "abc123")).to_tuple(),
            module_row(module="mod2", path="mod2.py", snapshot=("test/repo", "abc123")).to_tuple(),
        ],
    )

    result = fetch_paginated(
        con,
        "SELECT module FROM core.modules WHERE repo = ? ORDER BY module LIMIT ?",
        ["test/repo"],
        limit=10,
    )

    expect_is_instance(result, PaginatedRows)
    expected_count = 2
    expect_length(result.rows, expected_count)
    expect_false(result.truncated)


def test_row_exists_returns_true_when_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_exists returns True when at least one row matches."""
    con = fresh_gateway.con

    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            module_row(
                module="test_mod", path="test.py", snapshot=("test/repo", "abc123")
            ).to_tuple()
        ],
    )

    result = row_exists(
        con,
        "SELECT 1 FROM core.modules WHERE repo = ?",
        ["test/repo"],
    )

    expect_true(result)


def test_row_exists_returns_false_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_exists returns False when no rows match."""
    con = fresh_gateway.con

    result = row_exists(
        con,
        "SELECT 1 FROM core.modules WHERE repo = ?",
        ["nonexistent/repo"],
    )

    expect_false(result)


def test_paginated_rows_count_property() -> None:
    """Verify PaginatedRows.count returns length of rows."""
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    paginated = PaginatedRows(rows=rows, limit=10, truncated=False)

    expected_count = 3
    expect_equal(paginated.count, expected_count)


def test_base_repository_con_property(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository.con returns the underlying connection."""
    base_repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    expect_true(base_repo.con is fresh_gateway.con)


def test_base_repository_stores_attributes(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository stores gateway, repo, and commit attributes."""
    base_repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    expect_true(base_repo.gateway is fresh_gateway)
    expect_equal(base_repo.repo, "test/repo")
    expect_equal(base_repo.commit, "abc123")
