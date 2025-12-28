"""SQLGlot-based query builder for semantic specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.serving.semantic.models import FilterSpec, FilterValue
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterScalar


class SqlglotQueryBuilderError(ValueError):
    """Raised when a SQLGlot query cannot be built."""


def build_sqlglot_query(
    *,
    spec: SemanticQuerySpec,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> exp.Select:
    """Build a SQLGlot expression for a semantic query spec.

    Parameters
    ----------
    spec
        Semantic query spec to translate into SQLGlot expressions.
    allowed_columns
        Columns permitted for selection, filtering, and ordering.
    column_types
        Optional column type mapping for operator validation.

    Returns
    -------
    sqlglot.expressions.Select
        SQLGlot Select expression representing the query.
    """
    _validate_pagination(limit=spec.limit, offset=spec.offset)

    for col in spec.columns:
        _require_allowed_column(column=col, allowed_columns=allowed_columns, ctx="select")

    schema_name, table_name = split_table_key(spec.table_key)
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(schema_name),
    )

    select_exprs = [_column_expr(col) for col in spec.columns]
    expr = exp.select(*select_exprs).from_(table_expr)

    predicates = _build_predicates(
        filters=spec.filters,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if predicates is not None:
        expr = expr.where(predicates)

    if spec.order_by:
        expr = expr.order_by(
            *_order_by_exprs(spec.order_by, allowed_columns=allowed_columns),
        )

    if spec.limit or spec.offset:
        expr = expr.limit(spec.limit)
        if spec.offset:
            expr = expr.offset(spec.offset)

    return expr


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        msg = "limit must be >= 0"
        raise SqlglotQueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise SqlglotQueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise SqlglotQueryBuilderError(msg)


def _column_expr(column: str) -> exp.Column:
    return exp.Column(this=exp.to_identifier(column))


def _build_predicates(
    *,
    filters: list[FilterSpec],
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> exp.Expression | None:
    if not filters:
        return None
    predicates: list[exp.Expression] = []
    for filt in filters:
        _require_allowed_column(
            column=filt.column,
            allowed_columns=allowed_columns,
            ctx="filter",
        )
        predicates.append(_build_predicate(filt=filt, column_types=column_types))
    return _combine_predicates(predicates)


def _build_predicate(
    *, filt: FilterSpec, column_types: Mapping[str, ColumnType] | None
) -> exp.Expression:
    column_type = column_types.get(filt.column) if column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if filt.op not in allowed_ops:
        msg = (
            "Operator "
            f"{filt.op} is not supported for column type {column_type or _UNKNOWN_COLUMN_TYPE}"
        )
        raise SqlglotQueryBuilderError(msg)

    col_expr = _column_expr(filt.column)
    op = filt.op
    value = filt.value

    if op in _COMPARISON_OPS:
        return _build_comparison_predicate(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )
    if op == "in":
        return _build_in_predicate(col_expr=col_expr, value=value, column_type=column_type)
    if op in _STRING_OPS:
        return _build_string_predicate(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )

    msg = f"Unsupported operator: {op}"
    raise SqlglotQueryBuilderError(msg)


_UNKNOWN_COLUMN_TYPE = "UNKNOWN"
_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "lte", "gt", "gte"})
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


def _build_comparison_predicate(
    *,
    col_expr: exp.Column,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> exp.Expression:
    if isinstance(value, list):
        msg = f"{op} operator does not support list value"
        raise SqlglotQueryBuilderError(msg)
    if op in _ORDERING_OPS and column_type == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise SqlglotQueryBuilderError(msg)
    literal = _literal_expr(value)
    if op == "eq":
        return exp.EQ(this=col_expr, expression=literal)
    if op == "ne":
        return exp.NEQ(this=col_expr, expression=literal)
    if op == "lt":
        return exp.LT(this=col_expr, expression=literal)
    if op == "lte":
        return exp.LTE(this=col_expr, expression=literal)
    if op == "gt":
        return exp.GT(this=col_expr, expression=literal)
    if op == "gte":
        return exp.GTE(this=col_expr, expression=literal)
    msg = f"Unsupported comparison operator: {op}"
    raise SqlglotQueryBuilderError(msg)


def _build_in_predicate(
    *,
    col_expr: exp.Column,
    value: FilterValue,
    column_type: ColumnType | None,
) -> exp.Expression:
    if not isinstance(value, list):
        msg = "IN operator requires list value"
        raise SqlglotQueryBuilderError(msg)
    if column_type == "JSON":
        msg = "IN operator is not supported for JSON columns"
        raise SqlglotQueryBuilderError(msg)
    if not value:
        return exp.false()
    constants = [_literal_expr(item) for item in value]
    return exp.In(this=col_expr, expressions=constants)


def _build_string_predicate(
    *,
    col_expr: exp.Column,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> exp.Expression:
    if not isinstance(value, str):
        msg = f"{op} operator requires string value"
        raise SqlglotQueryBuilderError(msg)
    if column_type is not None and column_type != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise SqlglotQueryBuilderError(msg)
    func_name = "contains" if op == "contains" else "starts_with"
    return exp.Anonymous(this=func_name, expressions=[col_expr, exp.Literal.string(value)])


def _literal_expr(value: FilterScalar) -> exp.Expression:
    if isinstance(value, bool):
        return exp.true() if value else exp.false()
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    if isinstance(value, str):
        return exp.Literal.string(value)
    msg = f"Unsupported literal type: {type(value).__name__}"
    raise SqlglotQueryBuilderError(msg)


def _order_by_exprs(
    order_by: Sequence[str],
    *,
    allowed_columns: frozenset[str],
) -> list[exp.Expression]:
    order_exprs: list[exp.Expression] = []
    for item in order_by:
        descending = item.startswith("-")
        col_name = item[1:] if descending else item
        _require_allowed_column(column=col_name, allowed_columns=allowed_columns, ctx="order_by")
        order_exprs.append(exp.Ordered(this=_column_expr(col_name), desc=descending))
    return order_exprs


def _combine_predicates(predicates: Sequence[exp.Expression]) -> exp.Expression | None:
    if not predicates:
        return None
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = exp.and_(combined, predicate)
    return combined


__all__ = ["SqlglotQueryBuilderError", "build_sqlglot_query"]
