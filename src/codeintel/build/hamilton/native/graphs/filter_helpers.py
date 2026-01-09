"""Plan-based filter helpers for graph pipelines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.expr_vocab import Expression
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import materialize_plan


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
    return materialize_plan(plan, use_threads=True)


__all__ = ["plan_filter_or_fallback"]
