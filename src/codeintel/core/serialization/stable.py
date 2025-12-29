"""Deterministic serialization helpers for keys, hashes, and JSON payloads."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path

type JsonValue = str | int | float | bool | dict[str, JsonValue] | list[JsonValue] | None


def stable_stringify(value: object) -> str:
    """Return a deterministic string representation for hashing or cache keys.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    str
        Deterministic string representation.
    """
    json_value = stable_json_value(value)
    if isinstance(json_value, str):
        return json_value
    return json.dumps(
        json_value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_json_value(
    value: object,
    *,
    omit_none: bool = False,
    omit_private_fields: bool = False,
) -> JsonValue:
    """Convert a value into a deterministic JSON-compatible representation.

    Parameters
    ----------
    value
        Value to serialize.
    omit_none
        When True, drop fields or items with None values.
    omit_private_fields
        When True, omit dataclass fields that start with "_".

    Returns
    -------
    JsonValue
        JSON-compatible representation.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return stable_json_value(value.value, omit_none=omit_none, omit_private_fields=omit_private_fields)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_json_value(
            value,
            omit_none=omit_none,
            omit_private_fields=omit_private_fields,
        )
    if isinstance(value, Mapping):
        return _mapping_json_value(
            value,
            omit_none=omit_none,
            omit_private_fields=omit_private_fields,
        )
    if isinstance(value, (list, tuple)):
        return [
            stable_json_value(item, omit_none=omit_none, omit_private_fields=omit_private_fields)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        return _sorted_sequence_json_value(
            value,
            omit_none=omit_none,
            omit_private_fields=omit_private_fields,
        )
    return str(value)


def _mapping_json_value(
    value: Mapping[object, object],
    *,
    omit_none: bool,
    omit_private_fields: bool,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        json_value = stable_json_value(
            item,
            omit_none=omit_none,
            omit_private_fields=omit_private_fields,
        )
        if omit_none and json_value is None:
            continue
        result[str(key)] = json_value
    return result


def _sorted_sequence_json_value(
    values: Sequence[object] | set[object] | frozenset[object],
    *,
    omit_none: bool,
    omit_private_fields: bool,
) -> list[JsonValue]:
    json_values = [
        stable_json_value(item, omit_none=omit_none, omit_private_fields=omit_private_fields)
        for item in values
    ]
    json_values.sort(key=_json_sort_key)
    return json_values


def _json_sort_key(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _dataclass_json_value(
    value: object,
    *,
    omit_none: bool,
    omit_private_fields: bool,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for field in dataclasses.fields(value):
        name = field.name
        if omit_private_fields and name.startswith("_"):
            continue
        field_value = getattr(value, name)
        json_value = stable_json_value(
            field_value,
            omit_none=omit_none,
            omit_private_fields=omit_private_fields,
        )
        if omit_none and json_value is None:
            continue
        result[name] = json_value
    return result


__all__ = ["JsonValue", "stable_json_value", "stable_stringify"]
