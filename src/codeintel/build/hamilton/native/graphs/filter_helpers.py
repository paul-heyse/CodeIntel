"""Plan-based filter helpers for graph pipelines."""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from codeintel.build.tabular.compute_helpers import safe_filter_expr
from codeintel.build.tabular.expr_vocab import Expression
from codeintel.build.tabular.plan_ops import Plan, materialize_plan


def plan_filter_or_fallback(
    table: pa.Table,
    expr: Expression,
    *,
    fallback_mask: Callable[[pa.Table], pa.Array | pa.ChunkedArray],
) -> pa.Table:
    """Filter a table using Plan.filter with a safe_filter_expr fallback.

    Returns
    -------
    pyarrow.Table
        Filtered table, using the fallback mask if the plan fails.
    """
    try:
        return materialize_plan(Plan.table(table).filter(expr), use_threads=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return safe_filter_expr(table, expr, fallback_mask=fallback_mask)


__all__ = ["plan_filter_or_fallback"]
