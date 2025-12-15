"""Safe Ibis query builder for semantic queries.

All user-provided values are embedded as typed expression literals (never
interpolated into raw SQL strings). Identifiers are validated against an
explicit allowlist from the semantic registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Callable

    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.serving.semantic.models import FilterSpec


class QueryBuilderError(ValueError):
    """Raised when query construction fails."""


@dataclass(frozen=True, slots=True)
class SemanticQueryPlan:
    """Plan for building a safe semantic SELECT query.

    Parameters
    ----------
    table_key
        Fully qualified table/view name.
    columns
        Columns to select.
    allowed_columns
        Set of valid column names for validation.
    filters
        Filter specifications.
    order_by
        Order columns (prefix "-" for DESC).
    limit
        Maximum rows.
    offset
        Rows to skip.
    """

    table_key: str
    columns: list[str]
    allowed_columns: frozenset[str]
    filters: list[FilterSpec]
    order_by: list[str]
    limit: int
    offset: int


def _validate_pagination(*, limit: int, offset: int) -> None:
    """Validate pagination inputs.

    Raises
    ------
    QueryBuilderError
        If limit or offset is negative.
    """
    if limit < 0:
        msg = "limit must be >= 0"
        raise QueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise QueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise QueryBuilderError(msg)


def _resolve_table(*, ibis_con: DuckDBBackend, table_key: str) -> it.Table:
    try:
        schema_name, table_name = split_table_key(table_key)
    except ValueError as exc:
        raise QueryBuilderError(str(exc)) from exc
    return ibis_con.table(table_name, database=schema_name)


def _predicate_eq(col: it.Value, value: object) -> it.BooleanValue:
    return col == ibis.literal(value)


def _predicate_ne(col: it.Value, value: object) -> it.BooleanValue:
    return col != ibis.literal(value)


def _predicate_lt(col: it.Value, value: object) -> it.BooleanValue:
    return col < ibis.literal(value)


def _predicate_lte(col: it.Value, value: object) -> it.BooleanValue:
    return col <= ibis.literal(value)


def _predicate_gt(col: it.Value, value: object) -> it.BooleanValue:
    return col > ibis.literal(value)


def _predicate_gte(col: it.Value, value: object) -> it.BooleanValue:
    return col >= ibis.literal(value)


_SIMPLE_PREDICATES: dict[str, Callable[[it.Value, object], it.BooleanValue]] = {
    "eq": _predicate_eq,
    "ne": _predicate_ne,
    "lt": _predicate_lt,
    "lte": _predicate_lte,
    "gt": _predicate_gt,
    "gte": _predicate_gte,
}


_STRING_PREDICATES: dict[str, Callable[[it.StringColumn, str], it.BooleanValue]] = {
    "contains": lambda col, value: col.contains(value),
    "startswith": lambda col, value: col.startswith(value),
}


def _build_predicate(
    *,
    table: it.Table,
    allowed_columns: frozenset[str],
    filter_spec: FilterSpec,
) -> it.BooleanValue:
    _require_allowed_column(column=filter_spec.column, allowed_columns=allowed_columns, ctx="filter")
    col_expr = table[filter_spec.column]
    op = filter_spec.op
    value = filter_spec.value

    simple = _SIMPLE_PREDICATES.get(op)
    if simple is not None:
        return simple(col_expr, value)

    if op == "in":
        if not isinstance(value, list):
            msg = "IN operator requires list value"
            raise QueryBuilderError(msg)
        values = [ibis.literal(v) for v in value]
        return col_expr.isin(values)

    string_predicate = _STRING_PREDICATES.get(op)
    if string_predicate is not None:
        if not isinstance(value, str):
            msg = f"{op} operator requires string value"
            raise QueryBuilderError(msg)
        string_expr = cast("it.StringColumn", col_expr)
        return string_predicate(string_expr, value)

    msg = f"Unsupported operator: {op}"
    raise QueryBuilderError(msg)


def _build_order_by(
    *,
    expr: it.Table,
    allowed_columns: frozenset[str],
    order_by: list[str],
) -> list[it.Column]:
    order_parts: list[it.Column] = []
    for col in order_by:
        descending = col.startswith("-")
        col_name = col[1:] if descending else col
        _require_allowed_column(column=col_name, allowed_columns=allowed_columns, ctx="order_by")
        order_parts.append(expr[col_name].desc() if descending else expr[col_name].asc())
    return order_parts


def build_query(*, ibis_con: DuckDBBackend, plan: SemanticQueryPlan) -> it.Table:
    """Build an Ibis query expression for a semantic view.

    Parameters
    ----------
    ibis_con
        Ibis DuckDB backend bound to the serving connection.
    plan
        Resolved query plan.

    Returns
    -------
    it.Table
        Ibis table expression with filters, ordering, and pagination applied.

    Raises
    ------
    QueryBuilderError
        If any identifier is invalid or column not allowed.
    """
    _validate_pagination(limit=plan.limit, offset=plan.offset)

    for col in plan.columns:
        if col not in plan.allowed_columns:
            msg = f"Unknown column: {col}"
            raise QueryBuilderError(msg)

    table = _resolve_table(ibis_con=ibis_con, table_key=plan.table_key)
    predicates = [
        _build_predicate(table=table, allowed_columns=plan.allowed_columns, filter_spec=f)
        for f in plan.filters
    ]

    expr = table
    if predicates:
        expr = expr.filter(*predicates)

    if plan.order_by:
        expr = expr.order_by(
            _build_order_by(expr=expr, allowed_columns=plan.allowed_columns, order_by=plan.order_by)
        )

    expr = expr.select([expr[c] for c in plan.columns])
    return expr.limit(plan.limit, offset=plan.offset)


__all__ = ["QueryBuilderError", "SemanticQueryPlan", "build_query"]
