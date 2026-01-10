"""Plan-based filter helpers for graph pipelines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.expr_vocab import Expression
from codeintel.build.tabular.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan

_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


def plan_filter_or_fallback(
    table: pa.Table,
    expr: Expression,
) -> pa.Table:
    """Filter a table using Plan.filter.

    Returns
    -------
    pyarrow.Table
        Filtered table from the plan lane.
    """
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(filter_expr=expr),
    )
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            _INTERNAL_PLAN_TABLE_KEY,
            mode="tolerant",
            ordering=plan.ordering,
        ),
    )
    return result.good


__all__ = ["plan_filter_or_fallback"]
