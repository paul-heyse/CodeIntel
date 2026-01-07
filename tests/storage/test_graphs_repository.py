"""Tests for GraphRepository class.

This module tests GraphRepository-specific functionality. Base repository
helper tests (fetch_one_dict, fetch_all_dicts, etc.) are in
tests/storage/repositories/test_base.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.storage.repositories.base import BaseRepository
from codeintel.storage.repositories.graphs import GraphRepository
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_empty,
    expect_equal,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.context import make_storage_context
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    GoidRow,
    insert_rows,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


EXPECTED_COUNT_1 = 1
EXPECTED_COUNT_2 = 2
EXPECTED_GOID_CALLER = 1001
EXPECTED_GOID_CALLEE = 1002


def _graph_repo(gateway: StorageGateway, repo: str, commit: str) -> GraphRepository:
    context = make_storage_context(gateway, repo=repo, commit=commit)
    return GraphRepository(context=context)


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
    now = datetime.now(tz=UTC)

    insert_rows(
        fresh_gateway,
        [
            GoidRow(
                goid_h128=EXPECTED_GOID_CALLER,
                urn="urn:test:caller",
                repo=repo,
                commit=commit,
                rel_path="caller.py",
                kind="function",
                qualname="mod.caller",
                start_line=1,
                end_line=10,
                created_at=now,
            ),
            GoidRow(
                goid_h128=EXPECTED_GOID_CALLEE,
                urn="urn:test:callee",
                repo=repo,
                commit=commit,
                rel_path="callee.py",
                kind="function",
                qualname="mod.callee",
                start_line=1,
                end_line=10,
                created_at=now,
            ),
        ],
    )

    insert_rows(
        fresh_gateway,
        [
            CallGraphNodeRow(
                goid_h128=EXPECTED_GOID_CALLER,
                language="python",
                kind="function",
                arity=0,
                is_public=True,
                rel_path="caller.py",
            ),
            CallGraphNodeRow(
                goid_h128=EXPECTED_GOID_CALLEE,
                language="python",
                kind="function",
                arity=1,
                is_public=True,
                rel_path="callee.py",
            ),
        ],
    )

    insert_rows(
        fresh_gateway,
        [
            CallGraphEdgeRow(
                repo=repo,
                commit=commit,
                caller_goid_h128=EXPECTED_GOID_CALLER,
                callee_goid_h128=EXPECTED_GOID_CALLEE,
                callsite_path="caller.py",
                callsite_line=5,
                callsite_col=10,
                language="python",
                kind="direct",
                resolved_via="callee",
                confidence=1.0,
                evidence={},
            )
        ],
    )

    return EXPECTED_GOID_CALLER, EXPECTED_GOID_CALLEE


def test_graph_repository_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify GraphRepository is immutable."""
    repo = _graph_repo(fresh_gateway, "test/repo", "abc123")
    assert_frozen(repo, "repo", "other")


def test_graph_repository_inherits_base_repository(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify GraphRepository inherits from BaseRepository."""
    repo = _graph_repo(fresh_gateway, "test/repo", "abc123")
    expect_is_instance(repo, BaseRepository)
    expect_true(repo.con is fresh_gateway.con)


def test_get_outgoing_callgraph_neighbors_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors returns empty list when no data."""
    repo = _graph_repo(fresh_gateway, "test/repo", "abc123")
    result = repo.get_outgoing_callgraph_neighbors(EXPECTED_GOID_CALLER, limit=10)
    expect_empty(result)


def test_get_incoming_callgraph_neighbors_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_incoming_callgraph_neighbors returns empty list when no data."""
    repo = _graph_repo(fresh_gateway, "test/repo", "abc123")
    result = repo.get_incoming_callgraph_neighbors(EXPECTED_GOID_CALLEE, limit=10)
    expect_empty(result)


def test_get_outgoing_callgraph_neighbors_with_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors returns edges from caller."""
    repo_slug = "test/repo"
    commit = "abc123"
    caller_goid, callee_goid = _seed_call_graph_data(fresh_gateway, repo_slug, commit)

    graph_repo = _graph_repo(fresh_gateway, repo_slug, commit)
    result = graph_repo.get_outgoing_callgraph_neighbors(caller_goid, limit=10)

    expect_length(result, EXPECTED_COUNT_1)
    expect_equal(result[0]["caller_goid_h128"], caller_goid)
    expect_equal(result[0]["callee_goid_h128"], callee_goid)


def test_get_incoming_callgraph_neighbors_with_data(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_incoming_callgraph_neighbors returns edges to callee."""
    repo_slug = "test/repo"
    commit = "abc123"
    caller_goid, callee_goid = _seed_call_graph_data(fresh_gateway, repo_slug, commit)

    graph_repo = _graph_repo(fresh_gateway, repo_slug, commit)
    result = graph_repo.get_incoming_callgraph_neighbors(callee_goid, limit=10)

    expect_length(result, EXPECTED_COUNT_1)
    expect_equal(result[0]["caller_goid_h128"], caller_goid)
    expect_equal(result[0]["callee_goid_h128"], callee_goid)


def test_get_outgoing_callgraph_neighbors_filters_by_repo_commit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors filters by repo/commit."""
    _seed_call_graph_data(fresh_gateway, "repo1", "commit1")

    graph_repo = _graph_repo(fresh_gateway, "other/repo", "other_commit")
    result = graph_repo.get_outgoing_callgraph_neighbors(EXPECTED_GOID_CALLER, limit=10)
    expect_empty(result)


def test_get_outgoing_callgraph_neighbors_respects_limit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify get_outgoing_callgraph_neighbors respects limit parameter."""
    repo_slug = "test/repo"
    commit = "abc123"
    now = datetime.now(tz=UTC)

    insert_rows(
        fresh_gateway,
        [
            GoidRow(
                goid_h128=EXPECTED_GOID_CALLER,
                urn="urn:test:caller",
                repo=repo_slug,
                commit=commit,
                rel_path="caller.py",
                kind="function",
                qualname="mod.caller",
                start_line=1,
                end_line=10,
                created_at=now,
            ),
            CallGraphNodeRow(
                goid_h128=EXPECTED_GOID_CALLER,
                language="python",
                kind="function",
                arity=0,
                is_public=True,
                rel_path="caller.py",
            ),
        ],
    )

    for i in range(5):
        callee_goid = EXPECTED_GOID_CALLEE + i
        insert_rows(
            fresh_gateway,
            [
                GoidRow(
                    goid_h128=callee_goid,
                    urn=f"urn:test:callee{i}",
                    repo=repo_slug,
                    commit=commit,
                    rel_path=f"callee{i}.py",
                    kind="function",
                    qualname=f"mod.callee{i}",
                    start_line=1,
                    end_line=10,
                    created_at=now,
                ),
                CallGraphNodeRow(
                    goid_h128=callee_goid,
                    language="python",
                    kind="function",
                    arity=i + 1,
                    is_public=True,
                    rel_path=f"callee{i}.py",
                ),
                CallGraphEdgeRow(
                    repo=repo_slug,
                    commit=commit,
                    caller_goid_h128=EXPECTED_GOID_CALLER,
                    callee_goid_h128=callee_goid,
                    callsite_path="caller.py",
                    callsite_line=5 + i,
                    callsite_col=10,
                    language="python",
                    kind="direct",
                    resolved_via=f"callee{i}",
                    confidence=1.0,
                    evidence={},
                ),
            ],
        )

    graph_repo = _graph_repo(fresh_gateway, repo_slug, commit)
    result = graph_repo.get_outgoing_callgraph_neighbors(
        EXPECTED_GOID_CALLER, limit=EXPECTED_COUNT_2
    )

    expect_length(result, EXPECTED_COUNT_2)
