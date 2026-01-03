"""Type converters for serialization.

This module provides functions for converting Python types to
JSON-compatible values and back.
"""

from __future__ import annotations

import types
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Union, get_args, get_origin

# Type alias for JSON-compatible values
type JsonValue = str | int | float | bool | dict[str, JsonValue] | list[JsonValue] | None


def serialize_value(value: object) -> JsonValue:
    """Serialize a value to JSON-compatible type.

    Handles common Python types including:
    - Primitives (str, int, float, bool, None)
    - Enums (converted to value)
    - Paths (converted to string)
    - Datetime/date (converted to ISO format)
    - Dataclasses (recursively serialized)
    - Dicts and lists (recursively processed)

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    JsonValue
        JSON-compatible value.

    Examples
    --------
    >>> serialize_value(42)
    42
    >>> serialize_value(Path("/tmp/test"))
    '/tmp/test'
    >>> from datetime import datetime
    >>> serialize_value(datetime(2024, 1, 1, 12, 0))
    '2024-01-01T12:00:00'
    """
    # Handle primitives and None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Handle special types
    result = _serialize_special_type(value)
    if result is not None:
        return result

    # Handle containers
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, frozenset, set)):
        return [serialize_value(v) for v in value]

    # Fallback to string representation
    return str(value)


def _serialize_special_type(value: object) -> JsonValue | None:
    """Serialize special types (Enum, Path, datetime, dataclass).

    Parameters
    ----------
    value
        Value to check and serialize.

    Returns
    -------
    JsonValue | None
        Serialized value, or None if not a special type.
    """
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if is_dataclass(value) and not isinstance(value, type):
        return {k: serialize_value(v) for k, v in asdict(value).items()}

    return None


def deserialize_value(value: JsonValue, target_type: type | None = None) -> object:
    """Deserialize a JSON value to Python type.

    Parameters
    ----------
    value
        JSON-compatible value to deserialize.
    target_type
        Optional target type for conversion. If None, returns value as-is.

    Returns
    -------
    object
        Deserialized Python value.

    Examples
    --------
    >>> deserialize_value("2024-01-01T12:00:00", datetime)
    datetime.datetime(2024, 1, 1, 12, 0)
    >>> deserialize_value("/tmp/test", Path)
    PosixPath('/tmp/test')
    """
    if value is None or target_type is None:
        return value

    origin = get_origin(target_type)
    if origin is Union or origin is types.UnionType:
        args = get_args(target_type)
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _deserialize_typed_value(value, non_none[0])
        return value

    return _deserialize_typed_value(value, target_type)


def _deserialize_typed_value(value: JsonValue, target_type: type) -> object:
    """Deserialize value to specific type.

    Parameters
    ----------
    value
        JSON value to deserialize.
    target_type
        Target type to deserialize to.

    Returns
    -------
    object
        Deserialized value.
    """
    # Handle Path
    if target_type is Path or (isinstance(target_type, type) and issubclass(target_type, Path)):
        return Path(str(value))

    # Handle datetime
    if target_type is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)

    # Handle date
    if target_type is date and isinstance(value, str):
        return date.fromisoformat(value)

    # Handle Enum
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(value)

    # Return as-is for primitives and unknown types
    return value


def serialize_dataclass_to_dict(obj: object, *, omit_none: bool = False) -> dict[str, JsonValue]:
    """Serialize a dataclass to dictionary.

    Parameters
    ----------
    obj
        Dataclass instance to serialize.
    omit_none
        If True, omit fields with None values.

    Returns
    -------
    dict[str, JsonValue]
        Dictionary representation.

    Raises
    ------
    TypeError
        If obj is not a dataclass instance.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        msg = f"Expected dataclass instance, got {type(obj).__name__}"
        raise TypeError(msg)

    raw = asdict(obj)
    result: dict[str, JsonValue] = {}

    for key, value in raw.items():
        serialized = serialize_value(value)
        if omit_none and serialized is None:
            continue
        result[key] = serialized

    return result


__all__ = [
    "JsonValue",
    "deserialize_value",
    "serialize_dataclass_to_dict",
    "serialize_value",
]
