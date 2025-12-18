"""Safe Ibis query builder for semantic queries.

All user-provided identifiers are validated against an explicit allowlist from
the semantic registry. User-provided scalar values are parameterized via
``ibis.param(...)`` and bound at compile/execute time; large IN-lists are staged
via Arrow-backed memtables with explicit cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import ibis
import pyarrow as pa

from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.serving.semantic.templates import QueryTemplate
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterSpec
    from codeintel.serving.semantic.templates import BoundQuery


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


_IN_LIST_MEMTABLE_THRESHOLD = 500


@dataclass(frozen=True, slots=True)
class _PredicateContext:
    allowed_columns: frozenset[str]
    column_types: Mapping[str, ColumnType] | None
    temp_tables: list[str]
    params: dict[it.Expr, object]


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


def _predicate_eq(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col == param, param, value


def _predicate_ne(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col != param, param, value


def _predicate_lt(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col < param, param, value


def _predicate_lte(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col <= param, param, value


def _predicate_gt(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col > param, param, value


def _predicate_gte(col: it.Value, value: object) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col >= param, param, value


_SIMPLE_PREDICATES: dict[
    str, Callable[[it.Value, object], tuple[it.BooleanValue, it.Scalar, object]]
] = {
    "eq": _predicate_eq,
    "ne": _predicate_ne,
    "lt": _predicate_lt,
    "lte": _predicate_lte,
    "gt": _predicate_gt,
    "gte": _predicate_gte,
}


def _predicate_contains(
    col: it.StringColumn, value: str
) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col.contains(cast("it.StringScalar", param)), param, value


def _predicate_startswith(
    col: it.StringColumn, value: str
) -> tuple[it.BooleanValue, it.Scalar, object]:
    param = ibis.param(col.type())
    return col.startswith(cast("it.StringScalar", param)), param, value


_STRING_PREDICATES: dict[
    str, Callable[[it.StringColumn, str], tuple[it.BooleanValue, it.Scalar, object]]
] = {
    "contains": _predicate_contains,
    "startswith": _predicate_startswith,
}


def _build_predicate(
    *,
    table: it.Table,
    filter_spec: FilterSpec,
    ctx: _PredicateContext,
) -> it.BooleanValue:
    _require_allowed_column(
        column=filter_spec.column, allowed_columns=ctx.allowed_columns, ctx="filter"
    )
    col_expr = table[filter_spec.column]
    op = filter_spec.op
    value = filter_spec.value

    column_type = ctx.column_types.get(filter_spec.column) if ctx.column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if op not in allowed_ops:
        msg = f"Operator {op} is not supported for column type {column_type or 'UNKNOWN'}"
        raise QueryBuilderError(msg)

    simple = _SIMPLE_PREDICATES.get(op)
    if simple is not None:
        predicate, param, param_value = _build_simple_predicate(
            op=op,
            predicate=simple,
            col_expr=col_expr,
            value=value,
            column_type=column_type,
        )
        ctx.params[param] = param_value
        return predicate

    if op == "in":
        return _build_in_predicate(
            col_expr=col_expr,
            value=value,
            column_type=column_type,
            temp_tables=ctx.temp_tables,
        )

    string_predicate = _STRING_PREDICATES.get(op)
    if string_predicate is not None:
        predicate, param, param_value = _build_string_predicate(
            op=op,
            predicate=string_predicate,
            col_expr=col_expr,
            value=value,
            column_type=column_type,
        )
        ctx.params[param] = param_value
        return predicate

    msg = f"Unsupported operator: {op}"
    raise QueryBuilderError(msg)


def _build_simple_predicate(
    *,
    op: str,
    predicate: Callable[[it.Value, object], tuple[it.BooleanValue, it.Scalar, object]],
    col_expr: it.Value,
    value: object,
    column_type: ColumnType | None,
) -> tuple[it.BooleanValue, it.Scalar, object]:
    if op in {"lt", "lte", "gt", "gte"} and column_type == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise QueryBuilderError(msg)
    return predicate(col_expr, value)


def _build_in_predicate(
    *,
    col_expr: it.Value,
    value: object,
    column_type: ColumnType | None,
    temp_tables: list[str],
) -> it.BooleanValue:
    if not isinstance(value, list):
        msg = "IN operator requires list value"
        raise QueryBuilderError(msg)
    if column_type == "JSON":
        msg = "IN operator is not supported for JSON columns"
        raise QueryBuilderError(msg)

    if len(value) >= _IN_LIST_MEMTABLE_THRESHOLD:
        staged = ibis.memtable(pa.table({"value": value}))
        name = getattr(staged.op(), "name", None)
        if not isinstance(name, str) or not name:
            msg = "Failed to stage IN-list values for query execution"
            raise QueryBuilderError(msg)
        temp_tables.append(name)
        return col_expr.isin(staged["value"])

    values = [ibis.literal(v, type=col_expr.type()) for v in value]
    return col_expr.isin(values)


def _build_string_predicate(
    *,
    op: str,
    predicate: Callable[[it.StringColumn, str], tuple[it.BooleanValue, it.Scalar, object]],
    col_expr: it.Value,
    value: object,
    column_type: ColumnType | None,
) -> tuple[it.BooleanValue, it.Scalar, object]:
    if not isinstance(value, str):
        msg = f"{op} operator requires string value"
        raise QueryBuilderError(msg)
    if column_type is not None and column_type != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise QueryBuilderError(msg)
    string_expr = cast("it.StringColumn", col_expr)
    return predicate(string_expr, value)


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


def build_query(
    *,
    ibis_con: DuckDBBackend,
    plan: SemanticQueryPlan,
    column_types: Mapping[str, ColumnType] | None = None,
) -> BoundQuery:
    """Build an Ibis query expression for a semantic view.

    Parameters
    ----------
    ibis_con
        Ibis DuckDB backend bound to the serving connection.
    plan
        Resolved query plan.
    column_types
        Optional mapping of column name to contract type. When provided, filter
        operators are validated against the contract types.

    Returns
    -------
    BoundQuery
        Bound query ready for compilation/execution, plus any staged temporary tables.

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
    temp_tables: list[str] = []
    params: dict[it.Expr, object] = {}
    ctx = _PredicateContext(
        allowed_columns=plan.allowed_columns,
        column_types=column_types,
        temp_tables=temp_tables,
        params=params,
    )
    predicates = [
        _build_predicate(
            table=table,
            filter_spec=f,
            ctx=ctx,
        )
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
    limited_expr = expr.limit(plan.limit, offset=plan.offset)
    template = QueryTemplate(expr=limited_expr, temp_tables=tuple(temp_tables))
    return template.bind(params)


__all__ = ["QueryBuilderError", "SemanticQueryPlan", "build_query"]
