"""Unified filter compiler for SQLGlot, DuckDB, Arrow, and Polars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.dataset as ds
from sqlglot import exp
from sqlglot.errors import SqlglotError

from codeintel.core.columnar.queryspec import (
    ProjectionSpec,
    QuerySpec,
    projection_spec_from_columns,
)
from codeintel.core.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    Expression,
    FunctionExpression,
    duckdb_type_for_column_type,
)
from codeintel.core.filters import (
    FilterOpError,
    FilterSpecLike,
    FilterValue,
    Op,
    allowed_ops_for_column_types,
    validate_filter_value,
)
from codeintel.core.schemas.primitives import ColumnType, column_type_base
from codeintel.core.serialization.json import normalize_duckdb_json_value
from codeintel.core.sqlglot_tools import canonicalize_expression_duckdb

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from polars import Expr as PolarsExpr

    from codeintel.core.filters import FilterScalar
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


@dataclass(frozen=True, slots=True)
class QuerySpecFilterRequest:
    """Input bundle for building a QuerySpec from filter specs."""

    filters: Sequence[FilterSpecLike]
    allowed_columns: frozenset[str]
    projection: ProjectionSpec | None = None
    columns: Sequence[str] | Mapping[str, ds.Expression] | None = None
    provenance_columns: Sequence[str] = ()
    column_types: Mapping[str, ColumnType] | None = None


def compile_filter_predicates(
    filters: Sequence[FilterSpecLike],
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
    allowed_ops_by_column = allowed_ops_for_column_types(column_types)
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
                allowed_ops=allowed_ops_by_column.get(filt.column),
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
    combined = _combine_sqlglot(expressions)
    if combined is None:
        return None
    try:
        return canonicalize_expression_duckdb(combined)
    except (SqlglotError, TypeError, ValueError):
        return combined


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


def arrow_predicate_from_filters(
    filters: Sequence[FilterSpecLike],
    *,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> ds.Expression | None:
    """Compile filter specs into an Arrow dataset predicate.

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
    pyarrow.dataset.Expression | None
        Dataset predicate expression.

    Raises
    ------
    FilterCompilerError
        Raised when filter compilation fails.
    """
    if not filters:
        return None
    try:
        predicates = compile_filter_predicates(
            filters,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )
    except FilterCompilerError as exc:
        raise FilterCompilerError(str(exc)) from exc
    return arrow_filter_expression(predicates)


def queryspec_from_filters(
    request: QuerySpecFilterRequest,
) -> QuerySpec | None:
    """Return a QuerySpec compiled from filter specs and projections.

    Parameters
    ----------
    request
        Bundled filter and projection parameters for QuerySpec creation.

    Returns
    -------
    QuerySpec | None
        QuerySpec when compilation succeeds; otherwise ``None``.

    Raises
    ------
    ValueError
        If both projection and columns are missing.
    """
    resolved_projection = request.projection
    if resolved_projection is None:
        if request.columns is None:
            msg = "QuerySpec creation requires projection or columns."
            raise ValueError(msg)
        resolved_projection = projection_spec_from_columns(
            request.columns,
            provenance_columns=request.provenance_columns,
        )
    try:
        predicate = arrow_predicate_from_filters(
            request.filters,
            allowed_columns=request.allowed_columns,
            column_types=request.column_types,
        )
    except FilterCompilerError:
        if request.filters:
            return None
        predicate = None
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=resolved_projection,
    )


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


def _duckdb_eq(column: Expression, literal: Expression) -> Expression:
    return column == literal


def _duckdb_ne(column: Expression, literal: Expression) -> Expression:
    return column != literal


def _duckdb_lt(column: Expression, literal: Expression) -> Expression:
    return column < literal


def _duckdb_lte(column: Expression, literal: Expression) -> Expression:
    return column <= literal


def _duckdb_gt(column: Expression, literal: Expression) -> Expression:
    return column > literal


def _duckdb_gte(column: Expression, literal: Expression) -> Expression:
    return column >= literal


_DUCKDB_COMPARISON_DISPATCH: dict[Op, Callable[[Expression, Expression], Expression]] = {
    "eq": _duckdb_eq,
    "ne": _duckdb_ne,
    "lt": _duckdb_lt,
    "lte": _duckdb_lte,
    "gt": _duckdb_gt,
    "gte": _duckdb_gte,
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
    column: Expression,
    literal: Expression,
) -> Expression:
    comparator = _DUCKDB_COMPARISON_DISPATCH.get(op)
    if comparator is None:
        msg = f"Unsupported comparison operator: {op}"
        raise FilterCompilerError(msg)
    return comparator(column, literal)


def _duckdb_in_predicate(*, column: Expression, predicate: FilterPredicate) -> Expression:
    values = _require_list_value(op=predicate.op, value=predicate.value)
    if not values:
        return ConstantExpression(0) == ConstantExpression(1)
    constants = [_duckdb_typed_constant(item, column_type=predicate.column_type) for item in values]
    return column.isin(*constants)


def _duckdb_string_predicate(*, column: Expression, predicate: FilterPredicate) -> Expression:
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
    "QuerySpecFilterRequest",
    "arrow_filter_expression",
    "arrow_predicate_from_filters",
    "compile_filter_predicates",
    "duckdb_filter_expression",
    "polars_filter_expression",
    "queryspec_from_filters",
    "sqlglot_filter_expression",
]
