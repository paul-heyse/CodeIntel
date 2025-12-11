"""Typed Ibis helpers for analytics computations."""

from __future__ import annotations

from collections.abc import Sequence

import ibis
import ibis.expr.types as it


def bool_and(*predicates: it.BooleanValue) -> it.BooleanValue:
    """
    Combine predicates with AND in a type-safe way.

    Returns
    -------
    BooleanValue
        Combined predicate.

    Raises
    ------
    ValueError
        If no predicates are provided.
    """
    if not predicates:
        msg = "At least one predicate is required"
        raise ValueError(msg)

    result = predicates[0]
    for predicate in predicates[1:]:
        result &= predicate
    return result


def bool_or(*predicates: it.BooleanValue) -> it.BooleanValue:
    """
    Combine predicates with OR in a type-safe way.

    Returns
    -------
    BooleanValue
        Combined predicate.

    Raises
    ------
    ValueError
        If no predicates are provided.
    """
    if not predicates:
        msg = "At least one predicate is required"
        raise ValueError(msg)

    result = predicates[0]
    for predicate in predicates[1:]:
        result |= predicate
    return result


def literal_sequence(values: Sequence[object]) -> it.Value:
    """
    Create a typed literal array for use in `.isin()` predicates.

    Returns
    -------
    Value
        Literal array expression.
    """
    return ibis.literal(list(values))


def zero_if_null(value: it.Value) -> it.Value:
    """
    Replace NULLs in integer expressions with zero.

    Returns
    -------
    Value
        Numeric expression with NULLs coalesced to zero.
    """
    return ibis.coalesce(value, ibis.literal(0, type="int64"))


def safe_ratio(
    numerator: it.Value,
    denominator: it.Value,
) -> it.Value:
    """
    Compute numerator/denominator with a zero-guard.

    Returns
    -------
    Value
        Ratio expression with zero-division guarded.
    """
    return (
        ibis.case()
        .when(denominator == 0, None)
        .else_(numerator.cast("float64") / denominator)
        .end()
    )


__all__ = [
    "bool_and",
    "bool_or",
    "literal_sequence",
    "safe_ratio",
    "zero_if_null",
]
