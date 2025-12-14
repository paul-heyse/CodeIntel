"""ID normalization utilities for DuckDB DECIMAL values.

This module provides the canonical functions for normalizing DuckDB
DECIMAL(38,0) values to Python integers. These functions are used
across the graphs and analytics packages.
"""

from __future__ import annotations

from decimal import Decimal
from typing import SupportsInt, cast


def normalize_decimal_id(value: object) -> int | None:
    """Normalize DuckDB DECIMAL(38,0) values to Python ints.

    This function handles the various representations that DuckDB may
    return for DECIMAL columns, including Decimal, int, bytes, and str.

    Parameters
    ----------
    value
        Value from DuckDB rows representing a GOID or other ID.

    Returns
    -------
    int | None
        Integer representation or None when parsing fails or value is None.

    Examples
    --------
    >>> from decimal import Decimal
    >>> normalize_decimal_id(Decimal("12345"))
    12345
    >>> normalize_decimal_id(None) is None
    True
    >>> normalize_decimal_id("99999")
    99999
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def as_int(value: object) -> int | None:
    """Coerce a value to int when possible.

    Handles int, Decimal, str, bytes, and bytearray types.

    Parameters
    ----------
    value
        Input value to convert via int().

    Returns
    -------
    int | None
        Converted int when coercion succeeds, otherwise None.

    Examples
    --------
    >>> as_int(42)
    42
    >>> as_int(b"9")
    9
    >>> as_int(None) is None
    True
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    result: int | None
    try:
        result = int(value) if isinstance(value, str) else int(cast("SupportsInt", value))
    except (TypeError, ValueError, OverflowError):
        result = None
    return result


__all__ = [
    "as_int",
    "normalize_decimal_id",
]
