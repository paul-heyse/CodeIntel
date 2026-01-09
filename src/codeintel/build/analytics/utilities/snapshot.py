"""Snapshot-scoped Plan helpers for analytics utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.plan_ops import Plan, materialize_plan

ORDER_ASC = "ascending"


def require_columns(table: pa.Table, columns: Sequence[str]) -> None:
    """Require that the provided columns exist on a table.

    Raises
    ------
    ValueError
        If any required columns are missing from the table.
    """
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        msg = f"Missing snapshot columns: {missing}"
        raise ValueError(msg)


def snapshot_plan(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    columns: Sequence[str] | None = None,
) -> Plan:
    """Build a Plan scoped to a repo/commit snapshot.

    Parameters
    ----------
    table
        Input table to scope.
    repo
        Optional repository identifier to filter on.
    commit
        Optional commit identifier to filter on.
    columns
        Optional column projection to apply after filtering.

    Returns
    -------
    Plan
        Plan filtered to the snapshot and optionally projected.
    """
    plan = Plan.table(table)
    if columns is not None:
        require_columns(table, columns)
    filters: list[Expression] = []
    if repo is not None and "repo" in table.column_names:
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in table.column_names:
        filters.append(E.field("commit") == E.scalar(commit))
    if filters:
        plan = plan.filter(E.and_(*filters))
    if columns is not None:
        plan = plan.project({name: E.field(name) for name in columns})
    return plan


def snapshot_table(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    columns: Sequence[str] | None = None,
    order_by: Sequence[str] | None = None,
) -> pa.Table:
    """Materialize a snapshot-scoped Plan with optional ordering.

    Returns
    -------
    pyarrow.Table
        Snapshot-scoped table, optionally ordered.
    """
    plan = snapshot_plan(table, repo=repo, commit=commit, columns=columns)
    if order_by:
        plan = plan.order_by(sort_keys=[(name, ORDER_ASC) for name in order_by])
    return materialize_plan(plan, use_threads=True)


__all__ = ["require_columns", "snapshot_plan", "snapshot_table"]
