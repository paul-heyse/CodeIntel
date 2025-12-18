"""Typed query-result coercion helpers.

DuckDB and Ibis return dynamically typed Python values for scalar queries
(`.fetchone()`, `.execute()`). This module provides runtime-checked coercion
helpers so call sites do not rely on unchecked casts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, cast

import ibis.expr.types as it

if TYPE_CHECKING:
    from typing import SupportsFloat, SupportsInt

__all__ = [
    "ScalarCoercionError",
    "ScalarExecution",
    "coerce_float",
    "coerce_int",
    "coerce_optional_float",
    "execute_float",
    "execute_int",
    "execute_optional_float",
]

_KIND_FLOAT = "float"
_KIND_INT = "int"


class ScalarExecution(Protocol):
    """Protocol for Ibis scalar expressions that support `.execute()`."""

    def execute(self, *, limit: object | None = None, **kwargs: object) -> object:
        """Execute the scalar expression and return a Python value."""
        ...

class ScalarCoercionError(TypeError):
    """Raised when a scalar query result cannot be coerced to the expected type."""

    def __init__(self, kind: str, *, ctx: str, value: object) -> None:
        message = f"Failed to coerce {ctx} to {kind}: {value!r} ({type(value).__name__})"
        super().__init__(message)
        self.kind = kind
        self.ctx = ctx
        self.raw_value = value


def coerce_int(value: object, *, ctx: str) -> int:
    """Coerce an arbitrary scalar value to an int with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB/Ibis.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    int
        Coerced integer value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to an integer.
    """
    if isinstance(value, bool):
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value)

    try:
        return int(cast("SupportsInt", value))
    except (TypeError, ValueError) as exc:
        raise ScalarCoercionError(_KIND_INT, ctx=ctx, value=value) from exc


def coerce_float(value: object, *, ctx: str) -> float:
    """Coerce an arbitrary scalar value to a float with runtime validation.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB/Ibis.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    float
        Coerced float value.

    Raises
    ------
    ScalarCoercionError
        If the value cannot be coerced to a float.
    """
    if isinstance(value, bool):
        raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value)
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError as exc:
            raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value) from exc

    try:
        return float(cast("SupportsFloat", value))
    except (TypeError, ValueError) as exc:
        raise ScalarCoercionError(_KIND_FLOAT, ctx=ctx, value=value) from exc


def coerce_optional_float(value: object | None, *, ctx: str) -> float | None:
    """Coerce a value to float, treating None/NaN as missing.

    Parameters
    ----------
    value
        Scalar value returned by DuckDB/Ibis.
    ctx
        Human-readable context string included in errors.

    Returns
    -------
    float | None
        Coerced float value, or None when the value is missing.
    """
    if value is None:
        return None
    coerced = coerce_float(value, ctx=ctx)
    return None if math.isnan(coerced) else coerced


def execute_int(expr: it.Value, *, ctx: str) -> int:
    """Execute an Ibis scalar expression and coerce the result to int.

    Parameters
    ----------
    expr
        Ibis scalar expression to execute.
    ctx
        Context string for errors.

    Returns
    -------
    int
        Coerced integer value.
    """
    raw = cast("ScalarExecution", expr).execute()
    return coerce_int(raw, ctx=ctx)


def execute_float(expr: it.Value, *, ctx: str) -> float:
    """Execute an Ibis scalar expression and coerce the result to float.

    Parameters
    ----------
    expr
        Ibis scalar expression to execute.
    ctx
        Context string for errors.

    Returns
    -------
    float
        Coerced float value.
    """
    raw = cast("ScalarExecution", expr).execute()
    return coerce_float(raw, ctx=ctx)


def execute_optional_float(expr: it.Value, *, ctx: str) -> float | None:
    """Execute an Ibis scalar expression and coerce the result to float|None.

    Parameters
    ----------
    expr
        Ibis scalar expression to execute.
    ctx
        Context string for errors.

    Returns
    -------
    float | None
        Coerced float value, or None when the value is missing.
    """
    raw = cast("ScalarExecution", expr).execute()
    return coerce_optional_float(raw, ctx=ctx)
