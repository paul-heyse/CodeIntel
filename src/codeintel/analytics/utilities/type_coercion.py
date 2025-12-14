"""Type coercion helpers for analytics pipelines.

This module provides safe type conversion functions that return None
when conversion is not possible, avoiding exceptions in data processing.
"""

from __future__ import annotations


def optional_str(value: object | None) -> str | None:
    """Return a string representation or None.

    Parameters
    ----------
    value
        Value to convert to string.

    Returns
    -------
    str | None
        Converted string or None when input is missing.
    """
    return str(value) if value is not None else None


def optional_int(value: object | None) -> int | None:  # noqa: PLR0911
    """Return an integer or None when value is not provided.

    Parameters
    ----------
    value
        Value to convert to integer.

    Returns
    -------
    int | None
        Converted integer or None when input is missing or invalid.
    """
    if value is None:
        return None
    # Handle bool first (before int check since bool is subclass of int)
    if isinstance(value, bool):
        return int(value)
    # Handle numeric types directly
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip()) if value.strip() else None
        except ValueError:
            return None
    # Handle Decimal and other numeric types via try/except
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def int_or_default(value: object | None, default: int = 0) -> int:
    """Return an integer, falling back to default when value is falsy.

    Parameters
    ----------
    value
        Value to convert to integer.
    default
        Default value to return when conversion fails.

    Returns
    -------
    int
        Integer value or default when empty.
    """
    converted = optional_int(value)
    return converted if converted is not None else default


def optional_float(value: object | None) -> float | None:  # noqa: PLR0911
    """Return a float or None when value is not provided.

    Parameters
    ----------
    value
        Value to convert to float.

    Returns
    -------
    float | None
        Converted float or None when input is missing or invalid.
    """
    if value is None:
        return None
    # Handle bool first (before int check since bool is subclass of int)
    if isinstance(value, bool):
        return float(int(value))
    # Handle numeric types directly
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip()) if value.strip() else None
        except ValueError:
            return None
    # Handle Decimal and other numeric types via try/except
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def optional_bool(value: object | None) -> bool | None:
    """Return a boolean or None when value is not provided.

    Parameters
    ----------
    value
        Value to convert to boolean.

    Returns
    -------
    bool | None
        Converted boolean or None when input is missing or invalid.
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
