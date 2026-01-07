"""Optional helpers for Arrow Acero execution plans."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

try:
    from pyarrow import acero
except ImportError:  # pragma: no cover - optional dependency
    acero = None


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
    decl = acero.Declaration.from_table(table)
    if filter_expr is not None:
        decl = acero.Declaration("filter", [decl], filter=filter_expr)
    decl = acero.Declaration("project", [decl], expressions=list(projections))
    decl = acero.Declaration(
        "aggregate",
        [decl],
        keys=list(keys),
        aggregates=list(aggregations),
    )
    return decl.to_table()


__all__ = [
    "build_exec_plan",
]
