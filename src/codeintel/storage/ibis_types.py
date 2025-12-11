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

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from ibis.expr.types import BooleanValue, Table

TableT = TypeVar("TableT", bound="Table")


def ibis_bool(expr: object) -> BooleanValue:
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
    return cast("BooleanValue", expr)


def ge(column: object, value: object) -> BooleanValue:
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
    return cast("BooleanValue", column >= value)  # type: ignore[operator]


def gt(column: object, value: object) -> BooleanValue:
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
    return cast("BooleanValue", column > value)  # type: ignore[operator]


def le(column: object, value: object) -> BooleanValue:
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
    return cast("BooleanValue", column <= value)  # type: ignore[operator]


def lt(column: object, value: object) -> BooleanValue:
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
    return cast("BooleanValue", column < value)  # type: ignore[operator]


def ne(column: object, value: object) -> BooleanValue:
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
    return cast("BooleanValue", column != value)  # type: ignore[operator]


def count_gt(expr: object, value: int) -> BooleanValue:
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
    return cast("BooleanValue", expr > value)  # type: ignore[operator]


def ilike(column: object, pattern: str) -> BooleanValue:
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
    return cast("BooleanValue", column.ilike(pattern))  # type: ignore[attr-defined]


def and_predicates(*predicates: object) -> BooleanValue:
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
        result &= ibis_bool(pred)  # type: ignore[assignment]
    return result


def or_predicates(*predicates: object) -> BooleanValue:
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
        result |= ibis_bool(pred)  # type: ignore[assignment]
    return result


def filter_by(table: TableT, *predicates: object) -> TableT:
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
    return table.filter(typed_predicates)  # type: ignore[return-value]


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
