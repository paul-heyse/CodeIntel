"""Generic dataclass serialization utilities.

Provide consistent serialization of result dataclasses to dictionaries
for JSON output.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

type JsonValue = str | int | float | bool | dict[str, JsonValue] | list[JsonValue] | None


def serialize_result(obj: object) -> dict[str, JsonValue]:
    """Serialize any dataclass to dict, handling nested types.

    Parameters
    ----------
    obj
        Dataclass instance to serialize.

    Returns
    -------
    dict[str, JsonValue]
        Dictionary representation of the dataclass.

    Raises
    ------
    TypeError
        If obj is not a dataclass instance.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        msg = f"Expected dataclass instance, got {type(obj).__name__}"
        raise TypeError(msg)
    raw = asdict(obj)
    return {k: _serialize_value(v) for k, v in raw.items()}


def _serialize_value(value: object) -> JsonValue:
    """Recursively serialize values to JSON-compatible types.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    JsonValue
        JSON-compatible value.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Path, datetime, date)):
        return value.isoformat() if isinstance(value, (datetime, date)) else str(value)

    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    return str(value)


__all__ = ["serialize_result"]
