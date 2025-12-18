"""Type-safe Ibis expression helpers for the build module.

This module is the build-local "escape hatch" for working with Ibis expressions in a way that
keeps the rest of ``src/codeintel/build`` free of high-friction ``cast(Any, ...)`` patterns.

Ibis uses operator overloading (e.g., ``table.col == value``) that returns expression objects,
but static type checkers cannot model these semantics precisely. These helpers provide small,
typed wrappers around common operations (predicates, filters, aggregations) while keeping the
minimum necessary casts centralized here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import ibis.expr.types as ir
from ibis import window

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TableT = TypeVar("TableT", bound=ir.Table)


def ibis_bool(expr: object) -> ir.BooleanValue:
    """Cast an Ibis comparison expression to ``ir.BooleanValue``."""

    return cast("ir.BooleanValue", expr)


def eq(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe equality comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) == value)
    return comparison


def ne(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe not-equal comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) != value)
    return comparison


def ge(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe greater-than-or-equal comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) >= value)
    return comparison


def gt(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe greater-than comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) > value)
    return comparison


def le(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe less-than-or-equal comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) <= value)
    return comparison


def lt(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe less-than comparison."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", column) < value)
    return comparison


def ilike(column: ir.Value, pattern: str) -> ir.BooleanValue:
    """Type-safe ILIKE pattern match."""

    return cast("ir.BooleanValue", cast("Any", column).ilike(pattern))


def count_gt(expr: ir.Value, value: int) -> ir.BooleanValue:
    """Type-safe count > value comparison (e.g., ``table.count() > 0``)."""

    comparison: ir.BooleanValue = cast("ir.BooleanValue", cast("Any", expr) > value)
    return comparison


def and_predicates(*predicates: object) -> ir.BooleanValue:
    """Combine multiple Ibis predicates with AND.

    Raises
    ------
    ValueError
        If no predicates are provided.
    """

    if not predicates:
        message = "At least one predicate is required"
        raise ValueError(message)

    result = ibis_bool(predicates[0])
    for predicate in predicates[1:]:
        result &= ibis_bool(predicate)
    return result


def or_predicates(*predicates: object) -> ir.BooleanValue:
    """Combine multiple Ibis predicates with OR.

    Raises
    ------
    ValueError
        If no predicates are provided.
    """

    if not predicates:
        message = "At least one predicate is required"
        raise ValueError(message)

    result = ibis_bool(predicates[0])
    for predicate in predicates[1:]:
        result |= ibis_bool(predicate)
    return result


def bool_not(expr: object) -> ir.BooleanValue:
    """Type-safe boolean negation for Ibis predicates."""

    return ~ibis_bool(expr)


def filter_by(table: TableT, *predicates: object) -> TableT:
    """Filter an Ibis table with type-safe predicates."""

    typed_predicates = [ibis_bool(predicate) for predicate in predicates]
    return cast("TableT", table.filter(typed_predicates))


def isin_values(column: ir.Value, values: Iterable[object]) -> ir.BooleanValue:
    """Type-safe ``isin`` helper."""

    return cast("ir.BooleanValue", cast("Any", column).isin(list(values)))


def is_null(column: ir.Value) -> ir.BooleanValue:
    """Type-safe ``isnull`` helper."""

    return cast("ir.BooleanValue", cast("Any", column).isnull())


def not_null(column: ir.Value) -> ir.BooleanValue:
    """Type-safe ``notnull`` helper."""

    return cast("ir.BooleanValue", cast("Any", column).notnull())


def fillna(expr: ir.Value, value: object) -> ir.Value:
    """Type-safe ``fillna`` helper."""

    return cast("ir.Value", cast("Any", expr).fillna(value))


def cast_dtype(expr: ir.Value, dtype: str) -> ir.Value:
    """Type-safe ``cast`` helper."""

    return cast("ir.Value", cast("Any", expr).cast(dtype))


def col_sum(expr: ir.Value) -> ir.Value:
    """Type-safe ``sum`` aggregator."""

    return cast("ir.Value", cast("Any", expr).sum())


def col_mean(expr: ir.Value) -> ir.Value:
    """Type-safe ``mean`` aggregator."""

    return cast("ir.Value", cast("Any", expr).mean())


def col_max(expr: ir.Value) -> ir.Value:
    """Type-safe ``max`` aggregator."""

    return cast("ir.Value", cast("Any", expr).max())


def col_min(expr: ir.Value) -> ir.Value:
    """Type-safe ``min`` aggregator."""

    return cast("ir.Value", cast("Any", expr).min())


def col_count(expr: ir.Value) -> ir.Value:
    """Type-safe ``count`` aggregator."""

    return cast("ir.Value", cast("Any", expr).count())


def col_nunique(expr: ir.Value) -> ir.Value:
    """Type-safe ``nunique`` aggregator."""

    return cast("ir.Value", cast("Any", expr).nunique())


def table_has_column(table: ir.Table, column: str) -> bool:
    """Return True when the table expression includes ``column``."""

    return column in cast("Any", table).columns


def add(left: ir.Value, right: object) -> ir.Value:
    """Type-safe addition for Ibis expressions."""

    return cast("ir.Value", cast("Any", left) + right)


def sub(left: ir.Value, right: object) -> ir.Value:
    """Type-safe subtraction for Ibis expressions."""

    return cast("ir.Value", cast("Any", left) - right)


def mul(left: ir.Value, right: object) -> ir.Value:
    """Type-safe multiplication for Ibis expressions."""

    return cast("ir.Value", cast("Any", left) * right)


def truediv(left: ir.Value, right: object) -> ir.Value:
    """Type-safe division for Ibis expressions."""

    return cast("ir.Value", cast("Any", left) / right)


def window_over(
    *,
    partition_by: Sequence[ir.Value] | None = None,
    order_by: Sequence[ir.Value | str] | None = None,
) -> object:
    """Create a typed window expression."""

    return window(group_by=list(partition_by or []), order_by=list(order_by or []))


__all__ = [
    "add",
    "and_predicates",
    "bool_not",
    "cast_dtype",
    "col_count",
    "col_max",
    "col_mean",
    "col_min",
    "col_nunique",
    "col_sum",
    "count_gt",
    "eq",
    "fillna",
    "filter_by",
    "ge",
    "gt",
    "ibis_bool",
    "ilike",
    "is_null",
    "isin_values",
    "le",
    "lt",
    "mul",
    "ne",
    "not_null",
    "or_predicates",
    "sub",
    "table_has_column",
    "truediv",
    "window_over",
]

