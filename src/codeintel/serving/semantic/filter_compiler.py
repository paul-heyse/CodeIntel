"""Unified filter compiler for SQLGlot, DuckDB, Arrow, and Polars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.dataset as ds
from sqlglot import exp

from codeintel.core.schemas.primitives import ColumnType, column_type_base
from codeintel.serving.semantic.filter_ops import FilterOpError, validate_filter_value
from codeintel.serving.semantic.models import FilterSpec, FilterValue, Op
from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    Expression,
    FunctionExpression,
    duckdb_type_for_column_type,
)
from codeintel.storage.helpers.json import normalize_duckdb_json_value

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from polars import Expr as PolarsExpr

    from codeintel.serving.semantic.models import FilterScalar
else:
    PolarsExpr = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


class FilterCompilerError(ValueError):
    """Raised when filters cannot be compiled."""


@dataclass(frozen=True, slots=True)
class FilterPredicate:
    """Validated filter predicate ready for compilation."""

    column: str
    op: Op
    value: FilterValue
    column_type: ColumnType | None


def compile_filter_predicates(
    filters: Sequence[FilterSpec],
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> tuple[FilterPredicate, ...]:
    """Validate and normalize filter specs into predicates.

    Parameters
    ----------
    filters
        Filter specifications to compile.
    allowed_columns
        Allowed column names for filtering.
    column_types
        Optional column type mapping for validation.

    Returns
    -------
    tuple[FilterPredicate, ...]
        Validated filter predicates.

    Raises
    ------
    FilterCompilerError
        If a filter column is unknown or filter values are invalid.
    """
    predicates: list[FilterPredicate] = []
    for filt in filters:
        if filt.column not in allowed_columns:
            msg = f"Unknown filter column: {filt.column}"
            raise FilterCompilerError(msg)
        column_type = column_types.get(filt.column) if column_types is not None else None
        try:
            value = validate_filter_value(
                op=filt.op,
                value=filt.value,
                column_type=column_type,
            )
        except FilterOpError as exc:
            raise FilterCompilerError(str(exc)) from exc
        predicates.append(
            FilterPredicate(
                column=filt.column,
                op=filt.op,
                value=value,
                column_type=column_type,
            )
        )
    return tuple(predicates)


def sqlglot_filter_expression(predicates: Sequence[FilterPredicate]) -> exp.Expression | None:
    """Compile filter predicates into a SQLGlot expression.

    Returns
    -------
    sqlglot.expressions.Expression | None
        Combined SQLGlot predicate, or None when no predicates exist.
    """
    expressions = [_sqlglot_predicate(pred) for pred in predicates]
    return _combine_sqlglot(expressions)


def duckdb_filter_expression(predicates: Sequence[FilterPredicate]) -> Expression | None:
    """Compile filter predicates into a DuckDB Expression API predicate.

    Returns
    -------
    Expression | None
        Combined DuckDB predicate, or None when no predicates exist.
    """
    expressions = [_duckdb_predicate(pred) for pred in predicates]
    return _combine_duckdb(expressions)


def arrow_filter_expression(predicates: Sequence[FilterPredicate]) -> ds.Expression | None:
    """Compile filter predicates into a PyArrow dataset expression.

    Returns
    -------
    pyarrow.dataset.Expression | None
        Combined Arrow predicate, or None when no predicates exist.
    """
    expressions = [_arrow_predicate(pred) for pred in predicates]
    return _combine_arrow([expr for expr in expressions if expr is not None])


def polars_filter_expression(predicates: Sequence[FilterPredicate]) -> PolarsExpr | None:
    """Compile filter predicates into a Polars expression.

    Returns
    -------
    polars.Expr | None
        Combined Polars predicate, or None when no predicates exist.
    """
    if pl is None:  # pragma: no cover
        return None
    expressions = [_polars_predicate(pred) for pred in predicates]
    return _combine_polars(expressions)


_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})

_DUCKDB_COMPARISON_DISPATCH: dict[Op, Callable[[ColumnExpression, Expression], Expression]] = {
    "eq": lambda column, literal: column == literal,
    "ne": lambda column, literal: column != literal,
    "lt": lambda column, literal: column < literal,
    "lte": lambda column, literal: column <= literal,
    "gt": lambda column, literal: column > literal,
    "gte": lambda column, literal: column >= literal,
}
_ARROW_COMPARISON_DISPATCH: dict[Op, Callable[[ds.Expression, FilterScalar], ds.Expression]] = {
    "eq": lambda field, value: field == value,
    "ne": lambda field, value: field != value,
    "lt": lambda field, value: field < value,
    "lte": lambda field, value: field <= value,
    "gt": lambda field, value: field > value,
    "gte": lambda field, value: field >= value,
}
_POLARS_COMPARISON_DISPATCH: dict[Op, Callable[[PolarsExpr, FilterScalar], PolarsExpr]] = {
    "eq": lambda expr, value: expr == value,
    "ne": lambda expr, value: expr != value,
    "lt": lambda expr, value: expr < value,
    "lte": lambda expr, value: expr <= value,
    "gt": lambda expr, value: expr > value,
    "gte": lambda expr, value: expr >= value,
}


def _sqlglot_predicate(predicate: FilterPredicate) -> exp.Expression:
    column_expr = exp.Column(this=exp.to_identifier(predicate.column))
    op = predicate.op
    value = predicate.value
    if op in _COMPARISON_OPS:
        literal = _sqlglot_literal(value)
        return _sqlglot_comparison(op=op, column=column_expr, literal=literal)
    if op == "in":
        if not isinstance(value, list):
            msg = "IN operator requires list value"
            raise FilterCompilerError(msg)
        if not value:
            return exp.false()
        literals = [_sqlglot_literal(item) for item in value]
        return exp.In(this=column_expr, expressions=literals)
    if op in _STRING_OPS:
        if not isinstance(value, str):
            msg = f"{op} operator requires string value"
            raise FilterCompilerError(msg)
        func_name = "contains" if op == "contains" else "starts_with"
        return exp.Anonymous(this=func_name, expressions=[column_expr, exp.Literal.string(value)])
    msg = f"Unsupported operator: {op}"
    raise FilterCompilerError(msg)


def _sqlglot_comparison(
    *,
    op: Op,
    column: exp.Column,
    literal: exp.Expression,
) -> exp.Expression:
    if op == "eq":
        return exp.EQ(this=column, expression=literal)
    if op == "ne":
        return exp.NEQ(this=column, expression=literal)
    if op == "lt":
        return exp.LT(this=column, expression=literal)
    if op == "lte":
        return exp.LTE(this=column, expression=literal)
    if op == "gt":
        return exp.GT(this=column, expression=literal)
    if op == "gte":
        return exp.GTE(this=column, expression=literal)
    msg = f"Unsupported comparison operator: {op}"
    raise FilterCompilerError(msg)


def _sqlglot_literal(value: FilterValue) -> exp.Expression:
    if isinstance(value, bool):
        return exp.true() if value else exp.false()
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    if isinstance(value, str):
        return exp.Literal.string(value)
    msg = f"Unsupported literal type: {type(value).__name__}"
    raise FilterCompilerError(msg)


def _duckdb_predicate(predicate: FilterPredicate) -> Expression:
    col_expr = ColumnExpression(predicate.column)
    op = predicate.op
    if op in _COMPARISON_OPS:
        literal = _duckdb_typed_constant(predicate.value, column_type=predicate.column_type)
        return _duckdb_comparison(op=op, column=col_expr, literal=literal)
    if op == "in":
        return _duckdb_in_predicate(column=col_expr, predicate=predicate)
    if op in _STRING_OPS:
        return _duckdb_string_predicate(column=col_expr, predicate=predicate)
    msg = f"Unsupported operator: {op}"
    raise FilterCompilerError(msg)


def _duckdb_comparison(
    *,
    op: Op,
    column: ColumnExpression,
    literal: Expression,
) -> Expression:
    comparator = _DUCKDB_COMPARISON_DISPATCH.get(op)
    if comparator is None:
        msg = f"Unsupported comparison operator: {op}"
        raise FilterCompilerError(msg)
    return comparator(column, literal)


def _duckdb_in_predicate(*, column: ColumnExpression, predicate: FilterPredicate) -> Expression:
    values = _require_list_value(op=predicate.op, value=predicate.value)
    if not values:
        return ConstantExpression(0) == ConstantExpression(1)
    constants = [_duckdb_typed_constant(item, column_type=predicate.column_type) for item in values]
    return column.isin(*constants)


def _duckdb_string_predicate(*, column: ColumnExpression, predicate: FilterPredicate) -> Expression:
    value = _require_string_value(op=predicate.op, value=predicate.value)
    literal = _duckdb_typed_constant(value, column_type=predicate.column_type)
    func_name = "contains" if predicate.op == "contains" else "starts_with"
    return FunctionExpression(func_name, column, literal)


def _require_list_value(*, op: Op, value: FilterValue) -> list[FilterScalar]:
    if isinstance(value, list):
        return value
    msg = f"{op.upper()} operator requires list value"
    raise FilterCompilerError(msg)


def _require_scalar_value(*, op: Op, value: FilterValue) -> FilterScalar:
    if not isinstance(value, list):
        return value
    msg = f"{op} operator does not support list value"
    raise FilterCompilerError(msg)


def _require_string_value(*, op: Op, value: FilterValue) -> str:
    if isinstance(value, str):
        return value
    msg = f"{op} operator requires string value"
    raise FilterCompilerError(msg)


def _duckdb_typed_constant(value: FilterValue, *, column_type: ColumnType | None) -> Expression:
    literal_value: object = value
    base = column_type_base(column_type) if column_type is not None else None
    if base == "JSON":
        literal_value = normalize_duckdb_json_value(value)
    literal = ConstantExpression(literal_value)
    if column_type is None:
        return literal
    if base in {"BOOLEAN", "INTEGER", "BIGINT", "DOUBLE", "VARCHAR"}:
        return literal
    duckdb_type = duckdb_type_for_column_type(column_type)
    if duckdb_type is None:
        return literal
    return literal.cast(duckdb_type)


def _arrow_predicate(predicate: FilterPredicate) -> ds.Expression | None:
    field = ds.field(predicate.column)
    op = predicate.op
    value = predicate.value
    result: ds.Expression | None = None
    if op in _COMPARISON_OPS:
        if not isinstance(value, list):
            result = _arrow_comparison(field, op=op, value=value)
    elif op == "in":
        values = value if isinstance(value, list) else [value]
        isin = getattr(field, "isin", None)
        if values and callable(isin):
            result = isin(values)
    elif op in _STRING_OPS:
        value_str = value if isinstance(value, str) else None
        method = _arrow_string_method(field, op=op) if value_str is not None else None
        if method is not None and value_str is not None:
            result = method(value_str)
    return result


def _arrow_comparison(field: ds.Expression, *, op: Op, value: FilterScalar) -> ds.Expression:
    comparator = _ARROW_COMPARISON_DISPATCH.get(op)
    if comparator is None:
        msg = f"Unsupported comparison operator: {op}"
        raise FilterCompilerError(msg)
    return comparator(field, value)


def _arrow_string_method(
    field: ds.Expression,
    *,
    op: Op,
) -> Callable[[str], ds.Expression] | None:
    if op == "contains":
        contains = getattr(field, "contains", None)
        return contains if callable(contains) else None
    if op == "startswith":
        starts_with = getattr(field, "starts_with", None)
        if callable(starts_with):
            return starts_with
        startswith = getattr(field, "startswith", None)
        return startswith if callable(startswith) else None
    return None


def _polars_predicate(predicate: FilterPredicate) -> PolarsExpr:
    if pl is None:  # pragma: no cover
        msg = "Polars is required for Polars filter compilation"
        raise FilterCompilerError(msg)
    col_expr = pl.col(predicate.column)
    op = predicate.op
    if op in _COMPARISON_OPS:
        value = _require_scalar_value(op=op, value=predicate.value)
        return _polars_comparison(op=op, expr=col_expr, value=value)
    if op == "in":
        return _polars_in_predicate(expr=col_expr, predicate=predicate)
    if op in _STRING_OPS:
        return _polars_string_predicate(expr=col_expr, predicate=predicate)
    msg = f"Unsupported operator: {op}"
    raise FilterCompilerError(msg)


def _polars_comparison(*, op: Op, expr: PolarsExpr, value: FilterScalar) -> PolarsExpr:
    comparator = _POLARS_COMPARISON_DISPATCH.get(op)
    if comparator is None:
        msg = f"Unsupported comparison operator: {op}"
        raise FilterCompilerError(msg)
    return comparator(expr, value)


def _polars_in_predicate(*, expr: PolarsExpr, predicate: FilterPredicate) -> PolarsExpr:
    values = _require_list_value(op=predicate.op, value=predicate.value)
    return expr.is_in(values)


def _polars_string_predicate(*, expr: PolarsExpr, predicate: FilterPredicate) -> PolarsExpr:
    value = _require_string_value(op=predicate.op, value=predicate.value)
    if predicate.op == "contains":
        return expr.str.contains(value)
    return expr.str.starts_with(value)


def _combine_sqlglot(expressions: Sequence[exp.Expression]) -> exp.Expression | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined = exp.and_(combined, expr)
    return combined


def _combine_duckdb(expressions: Sequence[Expression]) -> Expression | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined &= expr
    return combined


def _combine_arrow(expressions: Sequence[ds.Expression]) -> ds.Expression | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined &= expr
    return combined


def _combine_polars(expressions: Sequence[PolarsExpr]) -> PolarsExpr | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined &= expr
    return combined


__all__ = [
    "FilterCompilerError",
    "FilterPredicate",
    "arrow_filter_expression",
    "compile_filter_predicates",
    "duckdb_filter_expression",
    "polars_filter_expression",
    "sqlglot_filter_expression",
]
