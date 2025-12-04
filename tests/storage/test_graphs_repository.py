"""Comprehensive tests for graphs repository.

This module tests GraphRepository and base repository helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.base import (
    BaseRepository,
    PaginatedRows,
    fetch_all_dicts,
    fetch_one_dict,
    fetch_paginated,
    row_exists,
)
from codeintel.storage.repositories.graphs import GraphRepository

# Test constants
EXPECTED_COUNT_0 = 0
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
EXPECTED_TOTAL_AVAILABLE = 100
EXPECTED_GOID_CALLER = 1001
EXPECTED_GOID_CALLEE = 1002
EXPECTED_GOID_OTHER = 1003


# =============================================================================
# Base Repository Helper Tests
# =============================================================================


def test_fetch_one_dict_returns_single_row(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_one_dict returns first row as dict."""
    # Insert test data
    fresh_gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["test_mod", "test.py", "test/repo", "abc123"],
    )

    result = fetch_one_dict(
        fresh_gateway.con,
        "SELECT module, path FROM core.modules WHERE repo = ?",
        ["test/repo"],
    )

    assert result is not None
    assert result["module"] == "test_mod"
    assert result["path"] == "test.py"


def test_fetch_one_dict_returns_none_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_one_dict returns None when no rows match."""
    result = fetch_one_dict(
        fresh_gateway.con,
        "SELECT * FROM core.modules WHERE repo = ?",
        ["nonexistent"],
    )
    assert result is None


def test_fetch_all_dicts_returns_list_of_dicts(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_all_dicts returns list of dictionaries."""
    # Insert test data
    for i in range(EXPECTED_COUNT_3):
        fresh_gateway.con.execute(
            "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
            [f"mod_{i}", f"mod_{i}.py", "test/repo", "abc123"],
        )

    result = fetch_all_dicts(
        fresh_gateway.con,
        "SELECT module FROM core.modules WHERE repo = ? ORDER BY module",
        ["test/repo"],
    )

    assert len(result) == EXPECTED_COUNT_3
    assert result[0]["module"] == "mod_0"
    assert result[1]["module"] == "mod_1"
    assert result[2]["module"] == "mod_2"


def test_fetch_all_dicts_returns_empty_list_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_all_dicts returns empty list when no rows match."""
    result = fetch_all_dicts(
        fresh_gateway.con,
        "SELECT * FROM core.modules WHERE repo = ?",
        ["nonexistent"],
    )
    assert result == []


def test_fetch_paginated_returns_paginated_rows(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_paginated returns PaginatedRows with correct metadata."""
    # Insert test data
    for i in range(5):
        fresh_gateway.con.execute(
            "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
            [f"mod_{i}", f"mod_{i}.py", "test/repo", "abc123"],
        )

    result = fetch_paginated(
        fresh_gateway.con,
        "SELECT module FROM core.modules WHERE repo = ? LIMIT ?",
        ["test/repo"],
        limit=EXPECTED_COUNT_3,
    )

    assert isinstance(result, PaginatedRows)
    assert len(result.rows) == EXPECTED_COUNT_3
    assert result.limit == EXPECTED_COUNT_3
    assert result.truncated is True  # More rows exist
    assert result.count == EXPECTED_COUNT_3


def test_fetch_paginated_not_truncated(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_paginated shows truncated=False when all rows fit."""
    # Insert fewer rows than limit
    fresh_gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["mod", "mod.py", "test/repo", "abc123"],
    )

    result = fetch_paginated(
        fresh_gateway.con,
        "SELECT module FROM core.modules WHERE repo = ? LIMIT ?",
        ["test/repo"],
        limit=10,
    )

    assert result.truncated is False
    assert result.count == EXPECTED_COUNT_1


def test_row_exists_returns_true_when_exists(fresh_gateway: StorageGateway) -> None:
    """Verify row_exists returns True when row exists."""
    fresh_gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["mod", "mod.py", "test/repo", "abc123"],
    )

    result = row_exists(
        fresh_gateway.con,
        "SELECT 1 FROM core.modules WHERE repo = ?",
        ["test/repo"],
    )
    assert result is True


def test_row_exists_returns_false_when_not_exists(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify row_exists returns False when no row exists."""
    result = row_exists(
        fresh_gateway.con,
        "SELECT 1 FROM core.modules WHERE repo = ?",
        ["nonexistent"],
    )
    assert result is False


# =============================================================================
# PaginatedRows Tests
# =============================================================================


def test_paginated_rows_count_property() -> None:
    """Verify PaginatedRows.count returns length of rows."""
    result = PaginatedRows(
        rows=[{"a": 1}, {"a": 2}],
        limit=10,
        truncated=False,
    )
    assert result.count == EXPECTED_COUNT_2


def test_paginated_rows_is_frozen() -> None:
    """Verify PaginatedRows is immutable."""
    result = PaginatedRows(
        rows=[{"a": 1}],
        limit=10,
        truncated=False,
    )
    with pytest.raises(AttributeError):
        result.limit = 20  # type: ignore[misc]


def test_paginated_rows_stores_total_available() -> None:
    """Verify PaginatedRows stores total_available."""
    result = PaginatedRows(
        rows=[{"a": 1}],
        limit=10,
        truncated=True,
        total_available=EXPECTED_TOTAL_AVAILABLE,
    )
    assert result.total_available == EXPECTED_TOTAL_AVAILABLE


# =============================================================================
# BaseRepository Tests
# =============================================================================


def test_base_repository_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository is immutable."""
    repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    with pytest.raises(AttributeError):
        repo.repo = "other"  # type: ignore[misc]


def test_base_repository_con_property(fresh_gateway: StorageGateway) -> None:
    """Verify BaseRepository.con returns gateway connection."""
    repo = BaseRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    assert repo.con is fresh_gateway.con


# =============================================================================
# GraphRepository Tests
# =============================================================================


def _seed_call_graph_data(
    fresh_gateway: StorageGateway,
    repo: str,
    commit: str,
) -> tuple[int, int]:
    """
    Seed call graph data for testing.

    Returns
    -------
    tuple[int, int]
        Tuple of (caller_goid, callee_goid).
    """
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    # Insert GOIDs for functions
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            EXPECTED_GOID_CALLER,
            "urn:test:caller",
            repo,
            commit,
            "caller.py",
            "python",
            "function",
            "mod.caller",
            1,
            10,
            now,
        ],
    )
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            EXPECTED_GOID_CALLEE,
            "urn:test:callee",
            repo,
            commit,
            "callee.py",
            "python",
            "function",
            "mod.callee",
            1,
            10,
            now,
        ],
    )

    # Insert call graph nodes
    fresh_gateway.graph.insert_call_graph_nodes([
        (EXPECTED_GOID_CALLER, "python", "function", 0, True, "caller.py"),
        (EXPECTED_GOID_CALLEE, "python", "function", 1, True, "callee.py"),
    ])

    # Insert call graph edge (caller -> callee)
    fresh_gateway.graph.insert_call_graph_edges([
        (
            repo,
            commit,
            EXPECTED_GOID_CALLER,
            EXPECTED_GOID_CALLEE,
            "caller.py",
            5,
            10,
            "python",
            "direct",
            "callee",
            1.0,
            "{}",
        )
    ])

    return EXPECTED_GOID_CALLER, EXPECTED_GOID_CALLEE


def test_graph_repository_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify GraphRepository is immutable."""
    repo = GraphRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    with pytest.raises(AttributeError):
        repo.repo = "other"  # type: ignore[misc]


def test_graph_repository_inherits_base_repository(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify GraphRepository inherits from BaseRepository."""
    repo = GraphRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    assert isinstance(repo, BaseRepository)
    assert repo.con is fresh_gateway.con


def test_get_outgoing_callgraph_neighbors_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors returns empty list when no data."""
    repo = GraphRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    result = repo.get_outgoing_callgraph_neighbors(EXPECTED_GOID_CALLER, limit=10)
    assert result == []


def test_get_incoming_callgraph_neighbors_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_incoming_callgraph_neighbors returns empty list when no data."""
    repo = GraphRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    result = repo.get_incoming_callgraph_neighbors(EXPECTED_GOID_CALLEE, limit=10)
    assert result == []


def test_get_outgoing_callgraph_neighbors_with_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors returns edges from caller."""
    repo_slug = "test/repo"
    commit = "abc123"
    caller_goid, callee_goid = _seed_call_graph_data(fresh_gateway, repo_slug, commit)

    graph_repo = GraphRepository(
        gateway=fresh_gateway,
        repo=repo_slug,
        commit=commit,
    )
    result = graph_repo.get_outgoing_callgraph_neighbors(caller_goid, limit=10)

    assert len(result) == EXPECTED_COUNT_1
    assert result[0]["caller_goid_h128"] == caller_goid
    assert result[0]["callee_goid_h128"] == callee_goid


def test_get_incoming_callgraph_neighbors_with_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_incoming_callgraph_neighbors returns edges to callee."""
    repo_slug = "test/repo"
    commit = "abc123"
    caller_goid, callee_goid = _seed_call_graph_data(fresh_gateway, repo_slug, commit)

    graph_repo = GraphRepository(
        gateway=fresh_gateway,
        repo=repo_slug,
        commit=commit,
    )
    result = graph_repo.get_incoming_callgraph_neighbors(callee_goid, limit=10)

    assert len(result) == EXPECTED_COUNT_1
    assert result[0]["caller_goid_h128"] == caller_goid
    assert result[0]["callee_goid_h128"] == callee_goid


def test_get_outgoing_callgraph_neighbors_filters_by_repo_commit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors filters by repo/commit."""
    # Seed data for one repo/commit
    _seed_call_graph_data(fresh_gateway, "repo1", "commit1")

    # Query for different repo/commit should return empty
    graph_repo = GraphRepository(
        gateway=fresh_gateway,
        repo="other/repo",
        commit="other_commit",
    )
    result = graph_repo.get_outgoing_callgraph_neighbors(EXPECTED_GOID_CALLER, limit=10)
    assert result == []


def test_get_outgoing_callgraph_neighbors_respects_limit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors respects limit parameter."""
    con = fresh_gateway.con
    repo_slug = "test/repo"
    commit = "abc123"
    now = datetime.now(tz=UTC)

    # Insert caller
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            EXPECTED_GOID_CALLER,
            "urn:test:caller",
            repo_slug,
            commit,
            "caller.py",
            "python",
            "function",
            "mod.caller",
            1,
            10,
            now,
        ],
    )
    fresh_gateway.graph.insert_call_graph_nodes([
        (EXPECTED_GOID_CALLER, "python", "function", 0, True, "caller.py"),
    ])

    # Insert multiple callees
    for i in range(5):
        callee_goid = EXPECTED_GOID_CALLEE + i
        con.execute(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                callee_goid,
                f"urn:test:callee{i}",
                repo_slug,
                commit,
                f"callee{i}.py",
                "python",
                "function",
                f"mod.callee{i}",
                1,
                10,
                now,
            ],
        )
        fresh_gateway.graph.insert_call_graph_nodes([
            (callee_goid, "python", "function", i + 1, True, f"callee{i}.py"),
        ])
        fresh_gateway.graph.insert_call_graph_edges([
            (
                repo_slug,
                commit,
                EXPECTED_GOID_CALLER,
                callee_goid,
                "caller.py",
                5 + i,
                10,
                "python",
                "direct",
                f"callee{i}",
                1.0,
                "{}",
            )
        ])

    graph_repo = GraphRepository(
        gateway=fresh_gateway,
        repo=repo_slug,
        commit=commit,
    )
    result = graph_repo.get_outgoing_callgraph_neighbors(
        EXPECTED_GOID_CALLER, limit=EXPECTED_COUNT_2
    )

    # Should only return 2 even though 5 exist
    assert len(result) == EXPECTED_COUNT_2
