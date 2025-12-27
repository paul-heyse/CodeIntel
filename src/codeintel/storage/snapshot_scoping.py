"""Snapshot scoping helpers (repo/commit filtering).

This module centralizes the "scope by snapshot when columns exist" behavior so
that Warehouse and repositories cannot drift in semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from duckdb import ColumnExpression, ConstantExpression

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBRelation

__all__ = ["RepoCommitScope", "maybe_scope_by_repo_commit", "maybe_scope_by_snapshot"]


class RepoCommitScope(Protocol):
    """Structural type describing the repo/commit snapshot identity."""

    @property
    def repo(self) -> str:
        """Repository identifier to scope by."""
        ...

    @property
    def commit(self) -> str:
        """Commit identifier to scope by."""
        ...


def maybe_scope_by_repo_commit(
    table: DuckDBRelation,
    *,
    repo: str,
    commit: str,
) -> DuckDBRelation:
    """Apply repo/commit filtering when the table contains snapshot columns.

    Parameters
    ----------
    table
        Input table expression.
    repo
        Repository identifier to filter by.
    commit
        Commit identifier to filter by.

    Returns
    -------
    TableT
        Filtered table when `repo` and `commit` columns exist, otherwise the
        original table expression.
    """
    names = set(table.columns)
    if "repo" in names and "commit" in names:
        return table.filter(
            (ColumnExpression("repo") == ConstantExpression(repo))
            & (ColumnExpression("commit") == ConstantExpression(commit))
        )
    return table


def maybe_scope_by_snapshot(
    table: DuckDBRelation,
    *,
    snapshot: RepoCommitScope,
) -> DuckDBRelation:
    """Apply snapshot scoping based on a structural repo/commit identity.

    Parameters
    ----------
    table
        Input table expression.
    snapshot
        Snapshot identity providing `repo` and `commit`.

    Returns
    -------
    TableT
        Filtered table when `repo` and `commit` columns exist, otherwise the
        original table expression.
    """
    return maybe_scope_by_repo_commit(table, repo=snapshot.repo, commit=snapshot.commit)
