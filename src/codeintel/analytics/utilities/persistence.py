"""Persistence utilities for analytics data operations.

This module provides common data structures used for scoped persistence
operations across analytics modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeleteScope:
    """Specification for scoped deletion before insert.

    Used to define the scope of data to delete before inserting new rows,
    typically scoped by repository and commit to support incremental updates.

    Attributes
    ----------
    repo
        Repository identifier for the deletion scope.
    commit
        Commit hash for the deletion scope.
    columns
        Optional explicit column names for the WHERE clause.

    Examples
    --------
    >>> scope = DeleteScope(repo="org/repo", commit="abc123")
    >>> scope.repo
    'org/repo'
    """

    repo: str
    commit: str
    columns: tuple[str, ...] | None = None


__all__ = ["DeleteScope"]
