"""Arrow worklist builders for parsing pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext

_FUNCTION_KINDS = ("function", "method")


def build_function_ast_worklist(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    """Build a deduped worklist for function AST parsing.

    Returns
    -------
    pyarrow.Table
        Worklist with one row per function goid.
    """
    required = (
        "goid_h128",
        "rel_path",
        "qualname",
        "start_line",
        "end_line",
        "kind",
    )
    if not set(required).issubset(frame.column_names):
        return pa.Table.from_pylist([])
    has_created_at = "created_at" in frame.column_names
    columns = (*required, "created_at") if has_created_at else tuple(required)
    plan = snapshot_plan(
        frame,
        columns=columns,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key="core.goids",
        ),
    )
    plan = plan.filter(E.in_("kind", _FUNCTION_KINDS))
    plan = plan.aggregate(
        keys=[E.field("goid_h128")],
        aggregates=_worklist_aggregates(include_created_at=has_created_at),
    )
    execution_ctx = resolve_execution_context(resolve_columnar_context(ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    table = reader_to_table(reader)
    if not has_created_at:
        table = table.append_column("created_at", pa.nulls(table.num_rows))
    return table


def build_module_ast_worklist(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    """Build a deduped worklist for module AST parsing.

    Returns
    -------
    pa.Table
        Worklist table for module parsing.
    """
    required = {"path", "module"}
    if not required.issubset(frame.column_names):
        return pa.Table.from_pylist([])
    columns = ["path", "module"]
    has_language = "language" in frame.column_names
    if has_language:
        columns.append("language")
    plan = snapshot_plan(
        frame,
        columns=tuple(columns),
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key="core.modules",
        ),
    )
    filters = [E.is_valid("path"), E.is_valid("module")]
    if has_language:
        filters.append(E.or_(E.is_null("language"), E.field("language") == E.scalar("python")))
    plan = plan.filter(E.and_(*filters))
    plan = plan.aggregate(
        keys=[E.field("path")],
        aggregates=(("module", "list", None, "modules"),),
    )
    execution_ctx = resolve_execution_context(resolve_columnar_context(ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def _worklist_aggregates(*, include_created_at: bool) -> Sequence[tuple[str, str, None, str]]:
    aggregates: list[tuple[str, str, None, str]] = [
        ("rel_path", "min", None, "rel_path"),
        ("qualname", "min", None, "qualname"),
        ("start_line", "min", None, "start_line"),
        ("end_line", "max", None, "end_line"),
    ]
    if include_created_at:
        aggregates.append(("created_at", "min", None, "created_at"))
    return tuple(aggregates)


__all__ = ["build_function_ast_worklist", "build_module_ast_worklist"]
