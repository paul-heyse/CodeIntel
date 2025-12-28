"""Polars query builder for semantic specs."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from polars import Expr, LazyFrame

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.semantic.models import FilterScalar, FilterSpec, FilterValue
    from codeintel.serving.semantic.specs import SemanticQuerySpec

    type PolarsExpr = Expr
    type PolarsLazyFrame = LazyFrame
else:
    type PolarsExpr = object
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


class PolarsQueryBuilderError(ValueError):
    """Raised when Polars query construction fails."""


def apply_query_spec(
    lazyframe: PolarsLazyFrame,
    *,
    spec: SemanticQuerySpec,
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None = None,
) -> PolarsLazyFrame:
    """Apply a semantic query spec to a Polars LazyFrame.

    Returns
    -------
    polars.LazyFrame
        The filtered, ordered, and sliced lazy frame.

    Raises
    ------
    PolarsQueryBuilderError
        If the spec is invalid or uses unsupported operators.
    """
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    _validate_pagination(limit=spec.limit, offset=spec.offset)

    for col in spec.columns:
        _require_allowed_column(column=col, allowed_columns=allowed_columns, ctx="select")

    filters = _build_filter_exprs(
        filters=spec.filters,
        allowed_columns=allowed_columns,
        column_types=column_types,
    )
    if filters is not None:
        lazyframe = lazyframe.filter(filters)

    if spec.order_by:
        lazyframe = lazyframe.sort(
            by=_order_by_columns(spec.order_by, allowed_columns=allowed_columns),
            descending=_order_by_descending(spec.order_by),
        )

    lazyframe = lazyframe.select(spec.columns)

    if spec.offset or spec.limit:
        lazyframe = lazyframe.slice(spec.offset, spec.limit)
    return lazyframe


def _validate_pagination(*, limit: int, offset: int) -> None:
    if limit < 0:
        msg = "limit must be >= 0"
        raise PolarsQueryBuilderError(msg)
    if offset < 0:
        msg = "offset must be >= 0"
        raise PolarsQueryBuilderError(msg)


def _require_allowed_column(*, column: str, allowed_columns: frozenset[str], ctx: str) -> None:
    if column not in allowed_columns:
        msg = f"Unknown {ctx} column: {column}"
        raise PolarsQueryBuilderError(msg)


def _build_filter_exprs(
    *,
    filters: list[FilterSpec],
    allowed_columns: frozenset[str],
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr | None:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    if not filters:
        return None
    exprs: list[PolarsExpr] = []
    for filt in filters:
        _require_allowed_column(
            column=filt.column,
            allowed_columns=allowed_columns,
            ctx="filter",
        )
        exprs.append(
            _build_filter_expr(
                filt=filt,
                column_types=column_types,
            )
        )
    combined = exprs[0]
    for expr in exprs[1:]:
        combined &= expr
    return combined


def _build_filter_expr(
    *,
    filt: FilterSpec,
    column_types: Mapping[str, ColumnType] | None,
) -> PolarsExpr:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    column_type = column_types.get(filt.column) if column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if filt.op not in allowed_ops:
        msg = (
            "Operator "
            f"{filt.op} is not supported for column type {column_type or _UNKNOWN_COLUMN_TYPE}"
        )
        raise PolarsQueryBuilderError(msg)

    col_expr = pl.col(filt.column)
    op = filt.op
    value = filt.value

    if op in _COMPARISON_BUILDERS:
        return _build_comparison_expr(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )
    if op == "in":
        return _build_in_expr(col_expr=col_expr, value=value)
    if op in _STRING_OPS:
        return _build_string_expr(
            col_expr=col_expr,
            op=op,
            value=value,
            column_type=column_type,
        )

    msg = f"Unsupported operator: {op}"
    raise PolarsQueryBuilderError(msg)


_UNKNOWN_COLUMN_TYPE = "UNKNOWN"
_COMPARISON_BUILDERS: dict[str, Callable[[PolarsExpr, FilterScalar], PolarsExpr]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
}
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


def _build_comparison_expr(
    *,
    col_expr: PolarsExpr,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> PolarsExpr:
    if isinstance(value, list):
        msg = f"{op} operator does not support list value"
        raise PolarsQueryBuilderError(msg)
    if op in _ORDERING_OPS and column_type == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise PolarsQueryBuilderError(msg)
    builder = _COMPARISON_BUILDERS[op]
    return builder(col_expr, value)


def _build_in_expr(*, col_expr: PolarsExpr, value: FilterValue) -> PolarsExpr:
    if pl is None:  # pragma: no cover
        msg = "polars is required for Polars query execution"
        raise PolarsQueryBuilderError(msg)
    if not isinstance(value, list):
        msg = "IN operator requires list value"
        raise PolarsQueryBuilderError(msg)
    if not value:
        return pl.lit(value=False)
    return col_expr.is_in(value)


def _build_string_expr(
    *,
    col_expr: PolarsExpr,
    op: str,
    value: FilterValue,
    column_type: ColumnType | None,
) -> PolarsExpr:
    if not isinstance(value, str):
        msg = f"{op} operator requires string value"
        raise PolarsQueryBuilderError(msg)
    if column_type is not None and column_type != "VARCHAR":
        msg = f"{op} operator is only supported for VARCHAR columns"
        raise PolarsQueryBuilderError(msg)
    if op == "contains":
        return col_expr.str.contains(value, literal=True)
    return col_expr.str.starts_with(value)


def _order_by_columns(order_by: list[str], *, allowed_columns: frozenset[str]) -> list[str]:
    columns: list[str] = []
    for item in order_by:
        descending = item.startswith("-")
        col_name = item[1:] if descending else item
        _require_allowed_column(column=col_name, allowed_columns=allowed_columns, ctx="order_by")
        columns.append(col_name)
    return columns


def _order_by_descending(order_by: list[str]) -> list[bool]:
    return [item.startswith("-") for item in order_by]


__all__ = ["PolarsQueryBuilderError", "apply_query_spec"]
