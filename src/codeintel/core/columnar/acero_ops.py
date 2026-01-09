"""Optional helpers for Arrow Acero execution plans.

Deprecated: use ``codeintel.core.columnar.plan_ops`` + ``arrowdsl.run_pipeline`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.expr_vocab import E

try:
    from pyarrow import acero
except ImportError:  # pragma: no cover - optional dependency
    acero = None

if TYPE_CHECKING or acero is not None:
    from codeintel.core.columnar.arrowdsl import ExecutionPlan
    from codeintel.core.columnar.execution_context import resolve_execution_context
    from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
else:

    class HashJoinSpec:
        """Placeholder for HashJoinSpec when pyarrow.acero is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = "pyarrow.acero is unavailable in this environment."
            raise RuntimeError(msg)

    class Plan:
        """Placeholder for Plan when pyarrow.acero is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = "pyarrow.acero is unavailable in this environment."
            raise RuntimeError(msg)


def build_exec_plan(
    table: pa.Table,
    *,
    filter_expr: pc.Expression | None,
    projections: Sequence[str],
    aggregations: Sequence[tuple[str, str]],
    keys: Sequence[str],
) -> pa.Table:
    """Build and execute an Acero plan for filter/project/aggregate.

    Parameters
    ----------
    table
        Input Arrow table.
    filter_expr
        Optional filter expression.
    projections
        Column names to project.
    aggregations
        Aggregation tuples for Acero aggregate node.
    keys
        Group-by keys for aggregation.

    Returns
    -------
    pyarrow.Table
        Resulting table from the execution plan.

    Raises
    ------
    RuntimeError
        Raised when pyarrow.acero is unavailable.
    """
    if acero is None:
        msg = "pyarrow.acero is unavailable in this environment."
        raise RuntimeError(msg)
    plan = Plan.table(table)
    if filter_expr is not None:
        plan = plan.filter(filter_expr)
    if projections:
        expressions = [E.field(name) for name in projections]
        plan = plan.project(expressions, names=list(projections))
    if aggregations:
        aggregate_specs = [
            (target, function, None, f"{target}_{function}")
            for target, function in aggregations
        ]
        key_exprs = [E.field(name) for name in keys]
        plan = plan.aggregate(keys=key_exprs, aggregates=aggregate_specs)
    exec_plan = ExecutionPlan.from_plan(plan)
    return exec_plan.to_table(ctx=resolve_execution_context(None))


__all__ = [
    "HashJoinSpec",
    "Plan",
    "build_exec_plan",
]
