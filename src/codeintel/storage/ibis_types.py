"""Type-safe Ibis expression helpers.

This module provides wrapper functions that help static type checkers
understand Ibis expression semantics. Ibis uses operator overloading
that returns expression objects (not Python bools), but Python's type
system doesn't understand this without explicit annotations.

The core issue is that when you write:

    table.filter(table.repo == "foo")

Static type checkers see `table.repo == "foo"` as returning `bool`
(Python's normal comparison semantics), but Ibis actually returns
`BooleanValue` (an Ibis expression type).

This module provides:
1. `ibis_bool()` - Cast comparison results to BooleanValue
2. `ge()`, `gt()`, `le()`, `lt()` - Type-safe comparison operators
3. Helper functions for common filter patterns

Example
-------
Before (type error):

    expr = table.filter(table.repo == self.repo)

After (type-safe):

    from codeintel.storage.ibis_types import ibis_bool
    expr = table.filter(ibis_bool(table.repo == self.repo))

Or using the combinator:

    from codeintel.storage.ibis_types import and_filter
    expr = and_filter(table, table.repo == self.repo, table.commit == self.commit)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import ibis.expr.types as it
from ibis import window


def ibis_bool(expr: object) -> it.BooleanValue:
    """Cast an Ibis comparison expression to BooleanValue.

    This is a type-casting helper that tells static type checkers
    the result of an Ibis column comparison is a BooleanValue,
    not a Python bool.

    Parameters
    ----------
    expr
        An Ibis comparison expression (e.g., `table.col == value`).

    Returns
    -------
    BooleanValue
        The same expression, cast to the correct Ibis type.

    Example
    -------
    >>> table = gateway.ibis.table("core.modules")
    >>> predicate = ibis_bool(table.repo == "my-repo")
    >>> filtered = table.filter(predicate)
    """
    return cast("it.BooleanValue", expr)


def ge(column: it.Value, value: object) -> it.BooleanValue:
    """Type-safe greater-than-or-equal comparison.

    Parameters
    ----------
    column
        Ibis column to compare.
    value
        Value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    column_expr = cast("Any", column)
    comparison: it.BooleanValue = cast("it.BooleanValue", column_expr >= value)
    return comparison


def gt(column: it.Value, value: object) -> it.BooleanValue:
    """Type-safe greater-than comparison.

    Parameters
    ----------
    column
        Ibis column to compare.
    value
        Value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    column_expr = cast("Any", column)
    comparison: it.BooleanValue = cast("it.BooleanValue", column_expr > value)
    return comparison


def le(column: it.Value, value: object) -> it.BooleanValue:
    """Type-safe less-than-or-equal comparison.

    Parameters
    ----------
    column
        Ibis column to compare.
    value
        Value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    column_expr = cast("Any", column)
    comparison: it.BooleanValue = cast("it.BooleanValue", column_expr <= value)
    return comparison


def lt(column: it.Value, value: object) -> it.BooleanValue:
    """Type-safe less-than comparison.

    Parameters
    ----------
    column
        Ibis column to compare.
    value
        Value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    column_expr = cast("Any", column)
    comparison: it.BooleanValue = cast("it.BooleanValue", column_expr < value)
    return comparison


def ne(column: it.Value, value: object) -> it.BooleanValue:
    """Type-safe not-equal comparison.

    Parameters
    ----------
    column
        Ibis column to compare.
    value
        Value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    column_expr = cast("Any", column)
    comparison: it.BooleanValue = cast("it.BooleanValue", column_expr != value)
    return comparison


def count_gt(expr: it.Value, value: int) -> it.BooleanValue:
    """Type-safe count > value comparison.

    Use this for expressions like `table.count() > 0`.

    Parameters
    ----------
    expr
        Ibis scalar expression (typically from `.count()`).
    value
        Integer value to compare against.

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    comparison: it.BooleanValue = cast("it.BooleanValue", cast("Any", expr) > value)
    return comparison


def ilike(column: it.Value, pattern: str) -> it.BooleanValue:
    """
    Type-safe ILIKE pattern match.

    Parameters
    ----------
    column
        Ibis string column.
    pattern
        SQL LIKE pattern (e.g., "%search%").

    Returns
    -------
    BooleanValue
        Ibis boolean expression.
    """
    return cast("it.BooleanValue", cast("Any", column).ilike(pattern))


def and_predicates(*predicates: object) -> it.BooleanValue:
    """Combine multiple Ibis predicates with AND.

    Parameters
    ----------
    *predicates
        Ibis comparison expressions to AND together.

    Returns
    -------
    BooleanValue
        Combined predicate.

    Raises
    ------
    ValueError
        If no predicates are provided.

    Examples
    --------
    >>> combined = and_predicates(
    ...     table.repo == "my-repo",
    ...     table.commit == "abc123",
    ... )
    """
    if not predicates:
        message = "At least one predicate is required"
        raise ValueError(message)

    result = ibis_bool(predicates[0])
    for pred in predicates[1:]:
        result &= ibis_bool(pred)
    return result


def bool_and(*predicates: it.BooleanValue) -> it.BooleanValue:
    """Alias for ``and_predicates`` for backward compatibility."""
    return and_predicates(*predicates)


def or_predicates(*predicates: object) -> it.BooleanValue:
    """Combine multiple Ibis predicates with OR.

    Parameters
    ----------
    *predicates
        Ibis comparison expressions to OR together.

    Returns
    -------
    BooleanValue
        Combined predicate.

    Raises
    ------
    ValueError
        If no predicates are provided.

    Examples
    --------
    >>> combined = or_predicates(
    ...     table.repo == "my-repo",
    ...     table.commit == "abc123",
    ... )
    """
    if not predicates:
        message = "At least one predicate is required"
        raise ValueError(message)

    result = ibis_bool(predicates[0])
    for pred in predicates[1:]:
        result |= ibis_bool(pred)
    return result


def filter_by[TableT: it.Table](table: TableT, *predicates: object) -> TableT:
    """Filter an Ibis table with type-safe predicates.

    This is a convenience wrapper around `table.filter()` that handles
    the type casting automatically.

    Parameters
    ----------
    table
        Ibis table to filter.
    *predicates
        Comparison expressions to filter by.

    Returns
    -------
    Table
        Filtered table (same type as input).

    Example
    -------
    >>> table = gateway.ibis.table("core.modules")
    >>> filtered = filter_by(
    ...     table,
    ...     table.repo == "my-repo",
    ...     table.commit == "abc123",
    ... )
    """
    typed_predicates = [ibis_bool(p) for p in predicates]
    return cast("TableT", table.filter(typed_predicates))


def col_sum(expr: it.Value) -> it.Value:
    """Type-safe sum aggregator.

    Returns
    -------
    it.Value
        Sum expression.
    """
    return cast("it.Value", cast("Any", expr).sum())


def col_mean(expr: it.Value) -> it.Value:
    """Type-safe mean aggregator.

    Returns
    -------
    it.Value
        Mean expression.
    """
    return cast("it.Value", cast("Any", expr).mean())


def col_max(expr: it.Value) -> it.Value:
    """Type-safe max aggregator.

    Returns
    -------
    it.Value
        Max expression.
    """
    return cast("it.Value", cast("Any", expr).max())


def col_min(expr: it.Value) -> it.Value:
    """Type-safe min aggregator.

    Returns
    -------
    it.Value
        Min expression.
    """
    return cast("it.Value", cast("Any", expr).min())


def col_count(expr: it.Value) -> it.Value:
    """Type-safe count aggregator.

    Returns
    -------
    it.Value
        Count expression.
    """
    return cast("it.Value", cast("Any", expr).count())


def col_nunique(expr: it.Value) -> it.Value:
    """Type-safe nunique aggregator.

    Returns
    -------
    it.Value
        Distinct count expression.
    """
    return cast("it.Value", cast("Any", expr).nunique())


def bool_not(expr: object) -> it.BooleanValue:
    """Type-safe boolean negation for Ibis predicates.

    Returns
    -------
    it.BooleanValue
        Negated boolean expression.
    """
    return ~ibis_bool(expr)


def isin_values(column: it.Value, values: Iterable[object]) -> it.BooleanValue:
    """Type-safe isin helper.

    Returns
    -------
    it.BooleanValue
        Predicate indicating membership in values.
    """
    return cast("it.BooleanValue", cast("Any", column).isin(list(values)))


def window_over(
    *,
    partition_by: Sequence[it.Value] | None = None,
    order_by: Sequence[it.Value | str] | None = None,
) -> object:
    """Create a typed window expression.

    Returns
    -------
    it.Window
        Window specification for subsequent operations.
    """
    return window(group_by=list(partition_by or []), order_by=list(order_by or []))


__all__ = [
    "and_predicates",
    "bool_and",
    "bool_not",
    "col_count",
    "col_max",
    "col_mean",
    "col_min",
    "col_nunique",
    "col_sum",
    "count_gt",
    "filter_by",
    "ge",
    "gt",
    "ibis_bool",
    "ilike",
    "isin_values",
    "le",
    "lt",
    "ne",
    "or_predicates",
    "window_over",
]
