"""Environment parsing helpers.

This module centralizes environment-variable parsing so all entrypoints interpret
settings consistently (CLI, serving, build, ingestion).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def is_set(name: str) -> bool:
    """Return True when an environment variable is present (even if empty).

    Parameters
    ----------
    name
        Environment variable name.

    Returns
    -------
    bool
        True when present in the environment mapping.
    """
    return name in os.environ


def get_str(name: str, *, default: str | None = None, strip: bool = True) -> str | None:
    """Return a string environment variable, or default when unset.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value to return when the variable is unset.
    strip
        Whether to strip surrounding whitespace from the value.

    Returns
    -------
    str | None
        Parsed string value when set, otherwise the default.
    """
    if name not in os.environ:
        return default
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() if strip else value


def get_bool(name: str, *, default: bool | None = None) -> bool | None:
    """Parse a boolean env var.

    Interprets common values:
    - true: 1/true/yes/on
    - false: 0/false/no/off
    - empty/unrecognized: raises ValueError (unless unset, in which case default is returned)

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value to return when unset.

    Returns
    -------
    bool | None
        Parsed boolean value, or the default when unset.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognized value.
    """
    raw = get_str(name, default=None, strip=True)
    if raw is None:
        return default
    if not raw:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False

    message = f"Invalid boolean value for {name}: {raw!r}"
    raise ValueError(message)


def get_int(
    name: str,
    *,
    default: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Parse an integer env var with optional bounds validation.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value to return when unset.
    min_value
        Minimum acceptable value (inclusive).
    max_value
        Maximum acceptable value (inclusive).

    Returns
    -------
    int | None
        Parsed integer value, or the default when unset.

    Raises
    ------
    ValueError
        If the variable is set to a non-integer or violates bounds.
    """
    raw = get_str(name, default=None)
    if raw is None or not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        message = f"Invalid integer value for {name}: {raw!r}"
        raise ValueError(message) from exc

    if min_value is not None and value < min_value:
        message = f"Value for {name} must be >= {min_value}, got {value}"
        raise ValueError(message)
    if max_value is not None and value > max_value:
        message = f"Value for {name} must be <= {max_value}, got {value}"
        raise ValueError(message)
    return value


def get_float(
    name: str,
    *,
    default: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Parse a float env var with optional bounds validation.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value to return when unset.
    min_value
        Minimum acceptable value (inclusive).
    max_value
        Maximum acceptable value (inclusive).

    Returns
    -------
    float | None
        Parsed float value, or the default when unset.

    Raises
    ------
    ValueError
        If the variable is set to a non-float or violates bounds.
    """
    raw = get_str(name, default=None)
    if raw is None or not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        message = f"Invalid float value for {name}: {raw!r}"
        raise ValueError(message) from exc

    if min_value is not None and value < min_value:
        message = f"Value for {name} must be >= {min_value}, got {value}"
        raise ValueError(message)
    if max_value is not None and value > max_value:
        message = f"Value for {name} must be <= {max_value}, got {value}"
        raise ValueError(message)
    return value


def get_path(
    name: str,
    *,
    default: Path | None = None,
    must_exist: bool = False,
) -> Path | None:
    """Parse a path env var into a resolved Path.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value to return when unset.
    must_exist
        When True, require the resolved path to exist.

    Returns
    -------
    Path | None
        Resolved path value, or the default when unset.

    Raises
    ------
    ValueError
        If must_exist is True and the resolved path does not exist.
    """
    raw = get_str(name, default=None)
    if raw is None or not raw:
        return default
    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        message = f"Path for {name} does not exist: {path}"
        raise ValueError(message)
    return path


def split_csv(raw: str | None) -> tuple[str, ...]:
    """Split comma-separated values into a normalized tuple.

    Parameters
    ----------
    raw
        Raw CSV string (comma-separated) or None.

    Returns
    -------
    tuple[str, ...]
        Parsed entries with surrounding whitespace removed.
    """
    if raw is None:
        return ()
    stripped = raw.strip()
    if not stripped:
        return ()
    return tuple(part.strip() for part in stripped.split(",") if part.strip())
