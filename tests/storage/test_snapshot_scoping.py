"""Tests for snapshot scoping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.snapshot_scoping import maybe_scope_by_repo_commit
from tests._helpers.assertions.expectation_assertions import expect_true

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


def test_maybe_scope_by_repo_commit_adds_filter_when_columns_exist(
    fresh_gateway: StorageGateway,
) -> None:
    """maybe_scope_by_repo_commit adds repo+commit filters when snapshot columns exist."""
    fresh_gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules)
        VALUES ('a/repo', 'c1', '[]'), ('b/repo', 'c2', '[]')
        """
    )
    table = fresh_gateway.relation_from_table_key("core.repo_map")
    scoped = maybe_scope_by_repo_commit(table, repo="a/repo", commit="c1")
    rows = scoped.df()
    expect_true(len(rows) == 1, message="scoped relation returns one row")
    expect_true(rows.loc[0, "repo"] == "a/repo", message="scoped repo matches")
    expect_true(rows.loc[0, "commit"] == "c1", message="scoped commit matches")


def test_maybe_scope_by_repo_commit_is_noop_when_columns_missing(
    fresh_gateway: StorageGateway,
) -> None:
    """maybe_scope_by_repo_commit is a no-op when repo/commit columns are absent."""
    table = fresh_gateway.relation_from_table_key("core.ast_nodes")
    scoped = maybe_scope_by_repo_commit(table, repo="a/repo", commit="c1")
    expect_true(scoped.columns == table.columns, message="unscoped columns unchanged")
