"""Tests for repositories/base.py module."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    fetch_all_dicts,
    fetch_one_dict,
    fetch_paginated,
    row_exists,
)


def test_fetch_one_dict_returns_mapping(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_one_dict returns single row as dict."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit)
        VALUES ('test_mod', 'test.py', 'test/repo', 'abc123')
        """
    )

    result = fetch_one_dict(
        con,
        "SELECT module, path FROM core.modules WHERE repo = ? AND commit = ?",
        ["test/repo", "abc123"],
    )

    assert result is not None
    assert isinstance(result, dict)
    assert result["module"] == "test_mod"
    assert result["path"] == "test.py"


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

    assert result is None


def test_fetch_all_dicts_returns_list(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_all_dicts returns all rows as list of dicts."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit) VALUES
            ('mod1', 'mod1.py', 'test/repo', 'abc123'),
            ('mod2', 'mod2.py', 'test/repo', 'abc123'),
            ('mod3', 'mod3.py', 'test/repo', 'abc123')
        """
    )

    result = fetch_all_dicts(
        con,
        "SELECT module, path FROM core.modules WHERE repo = ? ORDER BY module",
        ["test/repo"],
    )

    assert isinstance(result, list)
    expected_count = 3
    assert len(result) == expected_count
    assert result[0]["module"] == "mod1"
    assert result[1]["module"] == "mod2"
    assert result[2]["module"] == "mod3"


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

    assert isinstance(result, list)
    assert len(result) == 0


def test_fetch_paginated_detects_truncation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_paginated detects when more rows exist than limit."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit) VALUES
            ('mod1', 'mod1.py', 'test/repo', 'abc123'),
            ('mod2', 'mod2.py', 'test/repo', 'abc123'),
            ('mod3', 'mod3.py', 'test/repo', 'abc123'),
            ('mod4', 'mod4.py', 'test/repo', 'abc123'),
            ('mod5', 'mod5.py', 'test/repo', 'abc123')
        """
    )

    result = fetch_paginated(
        con,
        "SELECT module FROM core.modules WHERE repo = ? ORDER BY module LIMIT ?",
        ["test/repo"],
        limit=3,
    )

    assert isinstance(result, PaginatedRows)
    expected_count = 3
    assert len(result.rows) == expected_count
    assert result.limit == expected_count
    assert result.truncated is True


def test_fetch_paginated_no_truncation_when_under_limit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_paginated reports no truncation when fewer rows than limit."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit) VALUES
            ('mod1', 'mod1.py', 'test/repo', 'abc123'),
            ('mod2', 'mod2.py', 'test/repo', 'abc123')
        """
    )

    result = fetch_paginated(
        con,
        "SELECT module FROM core.modules WHERE repo = ? ORDER BY module LIMIT ?",
        ["test/repo"],
        limit=10,
    )

    assert isinstance(result, PaginatedRows)
    expected_count = 2
    assert len(result.rows) == expected_count
    assert result.truncated is False


def test_row_exists_returns_true_when_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_exists returns True when at least one row matches."""
    con = fresh_gateway.con

    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit)
        VALUES ('test_mod', 'test.py', 'test/repo', 'abc123')
        """
    )

    result = row_exists(
        con,
        "SELECT 1 FROM core.modules WHERE repo = ?",
        ["test/repo"],
    )

    assert result is True


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

    assert result is False


def test_paginated_rows_count_property() -> None:
    """Verify PaginatedRows.count returns length of rows."""
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    paginated = PaginatedRows(rows=rows, limit=10, truncated=False)

    expected_count = 3
    assert paginated.count == expected_count


def test_base_repository_con_property(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository.con returns the underlying connection."""
    base_repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    assert base_repo.con is fresh_gateway.con


def test_base_repository_stores_attributes(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository stores gateway, repo, and commit attributes."""
    base_repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    assert base_repo.gateway is fresh_gateway
    assert base_repo.repo == "test/repo"
    assert base_repo.commit == "abc123"
