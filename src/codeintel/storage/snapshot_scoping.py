"""Snapshot scoping helpers (repo/commit filtering).

This module centralizes the "scope by snapshot when columns exist" behavior so
that Warehouse and repositories cannot drift in semantics.
"""

from __future__ import annotations

import ibis.expr.types as it

from codeintel.storage.ibis_types import filter_by

__all__ = ["maybe_scope_by_repo_commit"]


def maybe_scope_by_repo_commit[TableT: it.Table](
    table: TableT,
    *,
    repo: str,
    commit: str,
) -> TableT:
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
    schema = table.schema()
    names = set(schema.keys())
    if "repo" in names and "commit" in names:
        return filter_by(table, table["repo"] == repo, table["commit"] == commit)
    return table
