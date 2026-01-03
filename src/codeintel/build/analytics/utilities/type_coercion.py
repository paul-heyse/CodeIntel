"""Deprecated wrapper for type coercion helpers.

Use codeintel.core.query_results instead.
"""

from __future__ import annotations

from codeintel.core.query_results import (
    ScalarCoercionError,
    coerce_optional_float,
    coerce_optional_int,
    coerce_str,
)


def optional_str(value: object | None) -> str | None:
    """Return a string representation or None.

    Returns
    -------
    str | None
        String value when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    try:
        return coerce_str(value, ctx="analytics.optional_str")
    except ScalarCoercionError:
        return str(value)


def optional_int(value: object | None) -> int | None:
    """Return an integer or None when value is not provided.

    Returns
    -------
    int | None
        Integer value when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return coerce_optional_int(value, ctx="analytics.optional_int")
    except ScalarCoercionError:
        return None


def int_or_default(value: object | None, default: int = 0) -> int:
    """Return an integer, falling back to default when value is falsy.

    Returns
    -------
    int
        Coerced integer or the default value.
    """
    converted = optional_int(value)
    return converted if converted is not None else default


def optional_float(value: object | None) -> float | None:
    """Return a float or None when value is not provided.

    Returns
    -------
    float | None
        Float value when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return coerce_optional_float(value, ctx="analytics.optional_float")
    except ScalarCoercionError:
        return None


def optional_bool(value: object | None) -> bool | None:
    """Return a boolean or None when value is not provided.

    Returns
    -------
    bool | None
        Boolean value when coercion succeeds, otherwise None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


__all__ = [
    "int_or_default",
    "optional_bool",
    "optional_float",
    "optional_int",
    "optional_str",
]
