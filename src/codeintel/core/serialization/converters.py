"""Type converters for serialization.

This module provides functions for converting Python types to
JSON-compatible values and back.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from codeintel.core.serialization.stable import JsonValue, stable_json_value


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

    # Handle optional types (e.g., str | None)
    origin = getattr(target_type, "__origin__", None)
    if origin is type(None):
        return None

    # Try to deserialize as special type
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
        serialized = stable_json_value(value)
        if omit_none and serialized is None:
            continue
        result[key] = serialized

    return result


__all__ = [
    "JsonValue",
    "deserialize_value",
    "serialize_dataclass_to_dict",
]
