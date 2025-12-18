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
    fresh_gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules)
        VALUES ('a/repo', 'c1', '[]'), ('b/repo', 'c2', '[]')
        """
    )
    table = fresh_gateway.ibis.table("core.repo_map")
    scoped = maybe_scope_by_repo_commit(table, repo="a/repo", commit="c1")
    sql = fresh_gateway.ibis.con.compile(scoped)
    expect_true("WHERE" in sql.upper(), message="scoped SQL contains WHERE")
    expect_true("repo" in sql, message="scoped SQL references repo")
    expect_true("commit" in sql, message="scoped SQL references commit")


def test_maybe_scope_by_repo_commit_is_noop_when_columns_missing(
    fresh_gateway: StorageGateway,
) -> None:
    table = fresh_gateway.ibis.table("core.ast_nodes")
    scoped = maybe_scope_by_repo_commit(table, repo="a/repo", commit="c1")
    sql = fresh_gateway.ibis.con.compile(scoped)
    expect_true("WHERE" not in sql.upper(), message="unscoped SQL has no WHERE")

