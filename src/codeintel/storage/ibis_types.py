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

from typing import Any, cast

import ibis.expr.types as it


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


def ilike(column: it.StringValue, pattern: str) -> it.BooleanValue:
    """Type-safe ILIKE pattern match.

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
    return cast("it.BooleanValue", column.ilike(pattern))


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
        # The & operator on BooleanValue returns BooleanValue
        result = cast("it.BooleanValue", result & ibis_bool(pred))
    return result


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
        # The | operator on BooleanValue returns BooleanValue
        result = cast("it.BooleanValue", result | ibis_bool(pred))
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


__all__ = [
    "and_predicates",
    "count_gt",
    "filter_by",
    "ge",
    "gt",
    "ibis_bool",
    "ilike",
    "le",
    "lt",
    "ne",
    "or_predicates",
]
