"""Snapshot-scoped Plan helpers for analytics utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.plan_ops import Plan, materialize_plan
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.queryspec import PROVENANCE_FIELDS, ProjectionSpec, QuerySpec


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


def build_snapshot_query_spec(
    *,
    base_cols: Sequence[str],
    repo: str | None = None,
    commit: str | None = None,
    computed: Sequence[tuple[str, Expression]] = (),
    table: pa.Table | None = None,
) -> QuerySpec:
    """Build a QuerySpec scoped to a repo/commit snapshot.

    Returns
    -------
    QuerySpec
        Snapshot-scoped query specification with optional projection.
    """
    if table is not None:
        require_columns(table, base_cols)
        available = set(table.column_names)
    else:
        available = None
    predicate = _snapshot_predicate(
        available=available,
        repo=repo,
        commit=commit,
    )
    projection = ProjectionSpec(
        base_cols=tuple(base_cols),
        computed=tuple(computed),
    )
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def snapshot_plan(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    columns: Sequence[str] | None = None,
    ctx: ExecutionContext | None = None,
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
    ctx
        Optional execution context to determine provenance inclusion.

    Returns
    -------
    Plan
        Plan filtered to the snapshot and optionally projected.
    """
    base_cols = tuple(columns or ())
    spec = build_snapshot_query_spec(
        base_cols=base_cols,
        repo=repo,
        commit=commit,
        table=table,
    )
    plan = Plan.table(table)
    if spec.predicate is not None:
        plan = plan.filter(spec.predicate)
    projection = spec.project_expressions(provenance=_include_provenance(table, ctx=ctx))
    if projection:
        plan = plan.project(projection)
    return plan


def snapshot_table(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    columns: Sequence[str] | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Materialize a snapshot-scoped Plan.

    Returns
    -------
    pyarrow.Table
        Snapshot-scoped table.
    """
    plan = snapshot_plan(
        table,
        repo=repo,
        commit=commit,
        columns=columns,
        ctx=ctx,
    )
    return materialize_plan(plan, use_threads=True)


def snapshot_reader(
    table: pa.Table,
    *,
    repo: str | None = None,
    commit: str | None = None,
    columns: Sequence[str] | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.RecordBatchReader:
    """Materialize a snapshot-scoped Plan as a reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Snapshot-scoped reader.
    """
    plan = snapshot_plan(
        table,
        repo=repo,
        commit=commit,
        columns=columns,
        ctx=ctx,
    )
    use_threads = ctx.resolve_use_threads() if ctx is not None else True
    return plan.to_reader(use_threads=use_threads)


def _snapshot_predicate(
    *,
    available: set[str] | None,
    repo: str | None,
    commit: str | None,
) -> Expression | None:
    filters: list[Expression] = []
    if repo is not None and (available is None or "repo" in available):
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and (available is None or "commit" in available):
        filters.append(E.field("commit") == E.scalar(commit))
    if not filters:
        return None
    return E.and_(*filters)


def _include_provenance(table: pa.Table, *, ctx: ExecutionContext | None) -> bool:
    if ctx is None or not ctx.provenance:
        return False
    column_names = set(table.column_names)
    return all(output_name in column_names for output_name, _source_name in PROVENANCE_FIELDS)

__all__ = [
    "build_snapshot_query_spec",
    "require_columns",
    "snapshot_plan",
    "snapshot_reader",
    "snapshot_table",
]
