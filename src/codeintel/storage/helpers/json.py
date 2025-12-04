"""JSON serialization helpers for DuckDB column values.

This module provides utilities for encoding and decoding JSON data stored
in DuckDB columns. DuckDB may return JSON data as strings, dicts, or lists
depending on the query context.

All functions handle None values gracefully and provide sensible defaults
on parse failures.
"""

from __future__ import annotations

import json

__all__ = [
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "encode_json_compact",
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
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


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
