"""Canonical value parsing for CLI parameters.

This module provides the single source of truth for parsing string values
into Python types. All CLI parameter parsing should use these functions
to ensure consistent behavior across:

- Handler parameter accessors (context.py)
- CLI argument coercion (introspection/params.py)
- Environment variable parsing (config/env.py, config/service.py)
- Dynamic operation parameters (handlers/ops.py)
"""

from __future__ import annotations

_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on", "y"})
_FALSY_VALUES = frozenset({"false", "0", "no", "off", "n"})


def parse_bool(value: str) -> bool:
    """Parse a string into a boolean.

    Accept common truthy/falsy string representations:
    - Truthy: "true", "1", "yes", "on", "y"
    - Falsy: "false", "0", "no", "off", "n"

    Parameters
    ----------
    value
        String value to parse.

    Returns
    -------
    bool
        Parsed boolean value.

    Examples
    --------
    >>> parse_bool("true")
    True
    >>> parse_bool("1")
    True
    >>> parse_bool("yes")
    True
    >>> parse_bool("false")
    False
    >>> parse_bool("0")
    False
    >>> parse_bool("no")
    False
    >>> parse_bool("invalid")
    False
    """
    return value.strip().lower() in _TRUTHY_VALUES


def parse_bool_or_none(value: str | None, *, default: bool | None = None) -> bool | None:
    """Parse a string into a boolean with None handling.

    Use for environment variables and optional boolean flags where
    distinguishing "not set" from "false" matters.

    Parameters
    ----------
    value
        String value to parse, or None.
    default
        Default value if parsing fails or value is None.

    Returns
    -------
    bool | None
        Parsed boolean, or default if value is None or unrecognized.

    Examples
    --------
    >>> parse_bool_or_none("true")
    True
    >>> parse_bool_or_none("false")
    False
    >>> parse_bool_or_none(None)
    >>> parse_bool_or_none(None, default=True)
    True
    >>> parse_bool_or_none("invalid", default=False)
    False
    """
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUTHY_VALUES:
        return True
    if lowered in _FALSY_VALUES:
        return False
    return default


def parse_cli_value(value: str) -> str | int | float | bool:
    """Parse a CLI parameter string into the appropriate Python type.

    Attempt to parse in order: bool, int, float, then fall back to string.
    This provides automatic type coercion for dynamic CLI parameters.

    Parameters
    ----------
    value
        Raw string value from CLI.

    Returns
    -------
    str | int | float | bool
        Parsed value in the most specific applicable type.

    Examples
    --------
    >>> parse_cli_value("true")
    True
    >>> parse_cli_value("false")
    False
    >>> parse_cli_value("42")
    42
    >>> parse_cli_value("3.14")
    3.14
    >>> parse_cli_value("hello")
    'hello'
    >>> parse_cli_value("123abc")
    '123abc'
    """
    lowered = value.lower()

    if lowered in _TRUTHY_VALUES:
        return True
    if lowered in _FALSY_VALUES:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def is_truthy_string(value: str) -> bool:
    """Check if a string represents a truthy value.

    Parameters
    ----------
    value
        String to check.

    Returns
    -------
    bool
        True if the string is a recognized truthy value.

    Examples
    --------
    >>> is_truthy_string("yes")
    True
    >>> is_truthy_string("no")
    False
    >>> is_truthy_string("maybe")
    False
    """
    return value.strip().lower() in _TRUTHY_VALUES


__all__ = [
    "is_truthy_string",
    "parse_bool",
    "parse_bool_or_none",
    "parse_cli_value",
]
