"""Tests for GraphRepository class.

This module tests GraphRepository-specific functionality. Base repository
helper tests (fetch_one_dict, fetch_all_dicts, etc.) are in
tests/storage/repositories/test_base.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.base import BaseRepository
from codeintel.storage.repositories.graphs import GraphRepository
from tests._helpers import assert_frozen

# Test constants
EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_GOID_CALLER = 1001
EXPECTED_GOID_CALLEE = 1002


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
    fresh_gateway.graph.insert_call_graph_nodes(
        [
            (EXPECTED_GOID_CALLER, "python", "function", 0, True, "caller.py"),
            (EXPECTED_GOID_CALLEE, "python", "function", 1, True, "callee.py"),
        ]
    )

    # Insert call graph edge (caller -> callee)
    fresh_gateway.graph.insert_call_graph_edges(
        [
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
        ]
    )

    return EXPECTED_GOID_CALLER, EXPECTED_GOID_CALLEE


def test_graph_repository_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify GraphRepository is immutable."""
    repo = GraphRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )
    assert_frozen(repo, "repo", "other")


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
    fresh_gateway.graph.insert_call_graph_nodes(
        [
            (EXPECTED_GOID_CALLER, "python", "function", 0, True, "caller.py"),
        ]
    )

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
        fresh_gateway.graph.insert_call_graph_nodes(
            [
                (callee_goid, "python", "function", i + 1, True, f"callee{i}.py"),
            ]
        )
        fresh_gateway.graph.insert_call_graph_edges(
            [
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
            ]
        )

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
