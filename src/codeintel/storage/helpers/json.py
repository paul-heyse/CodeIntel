"""JSON serialization helpers for DuckDB column values.

This module provides utilities for encoding and decoding JSON data stored
in DuckDB columns. DuckDB may return JSON data as strings, dicts, or lists
depending on the query context.

All functions handle None values gracefully and provide sensible defaults
on parse failures.
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "deserialize_str_tuple",
    "encode_json_compact",
    "normalize_duckdb_json_value",
    "serialize_str_sequence",
]


def decode_json(value: object) -> object:
    """Decode JSON from a DuckDB column value.

    Handle the various forms DuckDB might return JSON data: as a raw string,
    as an already-parsed dict/list, or as None.

    Parameters
    ----------
    value
        Raw value from DuckDB (may be str, dict, list, or None).

    Returns
    -------
    object
        Parsed JSON (dict or list), or empty list on failure.

    Examples
    --------
    >>> decode_json('{"key": "value"}')
    {'key': 'value'}
    >>> decode_json(None)
    []
    >>> decode_json([1, 2, 3])
    [1, 2, 3]
    """
    if value is None:
        return []
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return []

    parsed: object | None = None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None

    return parsed if isinstance(parsed, (dict, list)) else []


def decode_json_dict(value: object) -> dict[str, object]:
    """Decode JSON value expecting a dictionary result.

    Parameters
    ----------
    value
        Raw value from DuckDB column (may be str, dict, list, or None).

    Returns
    -------
    dict[str, object]
        Parsed dictionary, or empty dict if parsing fails or result is not a dict.

    Examples
    --------
    >>> decode_json_dict('{"key": "value"}')
    {'key': 'value'}
    >>> decode_json_dict(None)
    {}
    >>> decode_json_dict([1, 2, 3])
    {}
    """
    raw = decode_json(value)
    return raw if isinstance(raw, dict) else {}


def decode_json_list(value: object) -> list[object]:
    """Decode JSON value expecting a list result.

    Parameters
    ----------
    value
        Raw value from DuckDB column (may be str, dict, list, or None).

    Returns
    -------
    list[object]
        Parsed list, or empty list if parsing fails or result is not a list.

    Examples
    --------
    >>> decode_json_list("[1, 2, 3]")
    [1, 2, 3]
    >>> decode_json_list(None)
    []
    >>> decode_json_list('{"key": "value"}')
    []
    """
    raw = decode_json(value)
    return raw if isinstance(raw, list) else []


def encode_json_compact(value: object) -> str:
    """Encode value as compact JSON with no extra whitespace.

    Parameters
    ----------
    value
        Value to encode (must be JSON-serializable).

    Returns
    -------
    str
        JSON-encoded string with no whitespace separators.

    Examples
    --------
    >>> encode_json_compact({"key": "value"})
    '{"key":"value"}'
    >>> encode_json_compact([1, 2, 3])
    '[1,2,3]'
    """
    return json.dumps(value, separators=(",", ":"))


def normalize_duckdb_json_value(value: object) -> object:
    """Normalize a Python value for DuckDB JSON column insertion.

    DuckDB's JSON column type accepts JSON text for parameter binding. This
    helper centralizes conversion rules so callers do not implement bespoke
    `json.dumps(...)` logic.

    Parameters
    ----------
    value
        Value destined for a DuckDB JSON-typed column.

    Returns
    -------
    object
        JSON text (str) for container inputs, otherwise the original value.
    """
    if isinstance(value, set):
        return encode_json_compact(sorted(value))
    if isinstance(value, (dict, list, tuple)):
        return encode_json_compact(value)
    return value


def serialize_str_sequence(items: Sequence[str]) -> str:
    """Serialize a sequence of strings to compact JSON array.

    This is a convenience wrapper for encoding string sequences (like
    dataset names or target names) to JSON for storage in DuckDB columns.

    Parameters
    ----------
    items
        Sequence of strings to serialize.

    Returns
    -------
    str
        JSON-encoded array string.

    Examples
    --------
    >>> serialize_str_sequence(["a", "b", "c"])
    '["a","b","c"]'
    >>> serialize_str_sequence(())
    '[]'
    """
    return encode_json_compact(list(items))


def deserialize_str_tuple(raw: str | None) -> tuple[str, ...]:
    """Deserialize JSON array to string tuple.

    This is a convenience wrapper for decoding JSON arrays stored in
    DuckDB columns back to typed string tuples. Handles None and empty
    strings gracefully.

    Parameters
    ----------
    raw
        JSON-encoded array or None.

    Returns
    -------
    tuple[str, ...]
        Tuple of strings, empty if raw is None or empty.

    Examples
    --------
    >>> deserialize_str_tuple('["a","b"]')
    ('a', 'b')
    >>> deserialize_str_tuple(None)
    ()
    >>> deserialize_str_tuple("")
    ()
    """
    if not raw:
        return ()
    items = decode_json_list(raw)
    return tuple(str(x) for x in items)
