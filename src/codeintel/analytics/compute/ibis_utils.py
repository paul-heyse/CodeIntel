"""Typed Ibis helpers for analytics computations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

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


def zero_if_null(value: it.Value) -> it.NumericValue:
    """
    Replace NULLs in integer expressions with zero.

    Returns
    -------
    NumericValue
        Numeric expression with NULLs coalesced to zero.
    """
    coalesced = ibis.coalesce(cast("Any", value), ibis.literal(0, type="int64"))
    return cast("it.NumericValue", coalesced)


def safe_ratio(
    numerator: it.NumericValue,
    denominator: it.NumericValue,
) -> it.NumericValue:
    """
    Compute numerator/denominator with a zero-guard.

    Returns
    -------
    Value
        Ratio expression with zero-division guarded.
    """
    num = numerator.cast("float64")
    denom = denominator.cast("float64")
    return ibis.cases(
        (denom == 0, None),
        else_=num / denom,
    )


__all__ = [
    "bool_and",
    "bool_or",
    "literal_sequence",
    "safe_ratio",
    "zero_if_null",
]
