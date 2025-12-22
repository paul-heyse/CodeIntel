"""Type-safe Ibis expression helpers.

This module is the shared "escape hatch" for working with Ibis expressions in a way that keeps
the rest of the codebase free of high-friction ``cast("Any", ...)`` patterns.

Ibis relies heavily on operator overloading (e.g., ``table.col == value``). Static type
checkers cannot model these semantics precisely, so call sites often end up with scattered
casts. Centralizing the minimal necessary casts here improves maintainability and hardens the
rest of the codebase against typing regressions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import ibis.expr.types as ir
from ibis import window


def get_column(table: ir.Table, name: str) -> ir.Value:
    """Return a typed column expression from an Ibis table.

    Parameters
    ----------
    table
        Ibis table expression.
    name
        Column name.

    Returns
    -------
    ir.Value
        Column expression.
    """
    return cast("ir.Value", table[name])


def ibis_bool(expr: object) -> ir.BooleanValue:
    """Cast an Ibis predicate expression to ``ir.BooleanValue``.

    Parameters
    ----------
    expr
        An Ibis predicate expression (e.g., ``table.col == value``).

    Returns
    -------
    ir.BooleanValue
        The same expression, typed as a boolean predicate.
    """
    return cast("ir.BooleanValue", expr)


def eq(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe equality comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Equality predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) == value)


def ne(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe not-equal comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Not-equal predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) != value)


def ge(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe greater-than-or-equal comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Greater-than-or-equal predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) >= value)


def gt(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe greater-than comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Greater-than predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) > value)


def le(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe less-than-or-equal comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Less-than-or-equal predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) <= value)


def lt(column: ir.Value, value: object) -> ir.BooleanValue:
    """Type-safe less-than comparison.

    Parameters
    ----------
    column
        Ibis column expression.
    value
        Value to compare against.

    Returns
    -------
    ir.BooleanValue
        Less-than predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column) < value)


def count_gt(expr: ir.Value, value: int) -> ir.BooleanValue:
    """Type-safe ``count > value`` comparison.

    Parameters
    ----------
    expr
        Ibis scalar expression (typically from ``.count()``).
    value
        Integer value to compare against.

    Returns
    -------
    ir.BooleanValue
        Greater-than predicate.
    """
    return cast("ir.BooleanValue", cast("Any", expr) > value)


def ilike(column: ir.Value, pattern: str) -> ir.BooleanValue:
    """Type-safe ILIKE pattern match.

    Parameters
    ----------
    column
        Ibis string expression.
    pattern
        SQL LIKE pattern (e.g., ``%search%``).

    Returns
    -------
    ir.BooleanValue
        ILIKE predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column).ilike(pattern))


def sort_desc(expr: ir.Value) -> ir.Column:
    """Return a descending sort key for an Ibis expression.

    Parameters
    ----------
    expr
        Ibis expression to sort.

    Returns
    -------
    ir.Column
        Sort key for descending order.
    """
    return cast("ir.Column", cast("Any", expr).desc())


def and_predicates(*predicates: object) -> ir.BooleanValue:
    """Combine multiple predicates with AND.

    Parameters
    ----------
    *predicates
        Ibis predicate expressions.

    Returns
    -------
    ir.BooleanValue
        Combined predicate.

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
    """Combine multiple predicates with OR.

    Parameters
    ----------
    *predicates
        Ibis predicate expressions.

    Returns
    -------
    ir.BooleanValue
        Combined predicate.

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
    """Type-safe boolean negation for Ibis predicates.

    Parameters
    ----------
    expr
        Ibis predicate expression.

    Returns
    -------
    ir.BooleanValue
        Negated predicate.
    """
    return ~ibis_bool(expr)


def filter_by[TableT: ir.Table](table: TableT, *predicates: object) -> TableT:
    """Filter a table with type-safe predicates.

    Parameters
    ----------
    table
        Ibis table expression.
    *predicates
        Ibis predicate expressions.

    Returns
    -------
    TableT
        Filtered table expression with the same type as the input.
    """
    typed_predicates = [ibis_bool(predicate) for predicate in predicates]
    return cast("TableT", table.filter(typed_predicates))


def isin_values(column: ir.Value, values: Iterable[object]) -> ir.BooleanValue:
    """Type-safe ``isin`` helper for membership in a Python iterable.

    Parameters
    ----------
    column
        Ibis value expression.
    values
        Values to test for membership.

    Returns
    -------
    ir.BooleanValue
        Membership predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column).isin(list(values)))


def isin_column(column: ir.Value, values: ir.Value) -> ir.BooleanValue:
    """Type-safe ``isin`` helper for membership in a column/subquery.

    Parameters
    ----------
    column
        Ibis value expression.
    values
        Column/subquery expression providing the membership values.

    Returns
    -------
    ir.BooleanValue
        Membership predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column).isin(values))


def is_null(column: ir.Value) -> ir.BooleanValue:
    """Type-safe ``isnull`` helper.

    Parameters
    ----------
    column
        Ibis value expression.

    Returns
    -------
    ir.BooleanValue
        Null predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column).isnull())


def not_null(column: ir.Value) -> ir.BooleanValue:
    """Type-safe ``notnull`` helper.

    Parameters
    ----------
    column
        Ibis value expression.

    Returns
    -------
    ir.BooleanValue
        Not-null predicate.
    """
    return cast("ir.BooleanValue", cast("Any", column).notnull())


def fillna(expr: ir.Value, value: object) -> ir.Value:
    """Type-safe ``fillna`` helper.

    Parameters
    ----------
    expr
        Ibis value expression.
    value
        Fill value.

    Returns
    -------
    ir.Value
        Value expression with nulls filled.
    """
    return cast("ir.Value", cast("Any", expr).fillna(value))


def cast_dtype(expr: ir.Value, dtype: str) -> ir.Value:
    """Type-safe ``cast`` helper.

    Parameters
    ----------
    expr
        Ibis value expression.
    dtype
        Target dtype string (Ibis backend-dependent).

    Returns
    -------
    ir.Value
        Value expression cast to the requested dtype.
    """
    return cast("ir.Value", cast("Any", expr).cast(dtype))


def col_sum(expr: ir.Value) -> ir.Value:
    """Type-safe ``sum`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Sum expression.
    """
    return cast("ir.Value", cast("Any", expr).sum())


def col_mean(expr: ir.Value) -> ir.Value:
    """Type-safe ``mean`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Mean expression.
    """
    return cast("ir.Value", cast("Any", expr).mean())


def col_max(expr: ir.Value) -> ir.Value:
    """Type-safe ``max`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Max expression.
    """
    return cast("ir.Value", cast("Any", expr).max())


def col_min(expr: ir.Value) -> ir.Value:
    """Type-safe ``min`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Min expression.
    """
    return cast("ir.Value", cast("Any", expr).min())


def col_count(expr: ir.Value) -> ir.Value:
    """Type-safe ``count`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Count expression.
    """
    return cast("ir.Value", cast("Any", expr).count())


def col_nunique(expr: ir.Value) -> ir.Value:
    """Type-safe ``nunique`` aggregator.

    Parameters
    ----------
    expr
        Ibis value expression.

    Returns
    -------
    ir.Value
        Distinct count expression.
    """
    return cast("ir.Value", cast("Any", expr).nunique())


def table_has_column(table: ir.Table, column: str) -> bool:
    """Return True when the table expression includes ``column``.

    Parameters
    ----------
    table
        Ibis table expression.
    column
        Column name.

    Returns
    -------
    bool
        True when the table has the requested column.
    """
    return column in cast("Any", table).columns


def add(left: ir.Value, right: object) -> ir.Value:
    """Type-safe addition for Ibis expressions.

    Parameters
    ----------
    left
        Ibis value expression.
    right
        Scalar or expression to add.

    Returns
    -------
    ir.Value
        Summed value expression.
    """
    return cast("ir.Value", cast("Any", left) + right)


def sub(left: ir.Value, right: object) -> ir.Value:
    """Type-safe subtraction for Ibis expressions.

    Parameters
    ----------
    left
        Ibis value expression.
    right
        Scalar or expression to subtract.

    Returns
    -------
    ir.Value
        Subtracted value expression.
    """
    return cast("ir.Value", cast("Any", left) - right)


def mul(left: ir.Value, right: object) -> ir.Value:
    """Type-safe multiplication for Ibis expressions.

    Parameters
    ----------
    left
        Ibis value expression.
    right
        Scalar or expression to multiply by.

    Returns
    -------
    ir.Value
        Multiplied value expression.
    """
    return cast("ir.Value", cast("Any", left) * right)


def truediv(left: ir.Value, right: object) -> ir.Value:
    """Type-safe division for Ibis expressions.

    Parameters
    ----------
    left
        Ibis value expression.
    right
        Scalar or expression to divide by.

    Returns
    -------
    ir.Value
        Divided value expression.
    """
    return cast("ir.Value", cast("Any", left) / right)


def window_over(
    *,
    partition_by: Sequence[ir.Value] | None = None,
    order_by: Sequence[ir.Value | str] | None = None,
) -> object:
    """Create a window specification for windowed operations.

    Parameters
    ----------
    partition_by
        Expressions to partition the window by.
    order_by
        Expressions or column names to order the window by.

    Returns
    -------
    object
        Window specification suitable for use with Ibis windowed operations.
    """
    return window(group_by=list(partition_by or []), order_by=list(order_by or []))


def select_columns(table: ir.Table, *columns: str) -> ir.Table:
    """Type-safe ``select`` wrapper for column name selection.

    Parameters
    ----------
    table
        Ibis table expression.
    *columns
        Column names to project.

    Returns
    -------
    ir.Table
        Projected table expression.
    """
    return cast("ir.Table", cast("Any", table).select(*columns))


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
    "get_column",
    "gt",
    "ibis_bool",
    "ilike",
    "is_null",
    "isin_column",
    "isin_values",
    "le",
    "lt",
    "mul",
    "ne",
    "not_null",
    "or_predicates",
    "select_columns",
    "sort_desc",
    "sub",
    "table_has_column",
    "truediv",
    "window_over",
]
