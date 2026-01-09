"""Plan-based filter helpers for graph pipelines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.expr_vocab import Expression
from codeintel.build.tabular.plan_ops import Plan, materialize_plan


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
    return materialize_plan(Plan.table(table).filter(expr), use_threads=True)


__all__ = ["plan_filter_or_fallback"]
