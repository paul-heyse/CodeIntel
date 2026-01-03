"""Boundary JSON helpers for DuckDB column values."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

import msgspec

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
    """
    if value is None:
        return []
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        decoded = _decode_bytes_payload(value)
        parsed = _coerce_decoded_payload(decoded)
        return parsed if isinstance(parsed, (dict, list)) else []
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
    """
    return json.dumps(value, separators=(",", ":"))


def _coerce_json_container(value: object) -> object:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _parse_json_value(raw: str) -> object | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None


def _decode_bytes_payload(value: bytes | bytearray | memoryview) -> object | None:
    raw = bytes(value)
    try:
        return msgspec.msgpack.decode(raw)
    except msgspec.DecodeError:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return _parse_json_value(text)


def _coerce_decoded_payload(value: object | None) -> object | None:
    if isinstance(value, str):
        parsed = _parse_json_value(value)
        return parsed if parsed is not None else value
    return value


def normalize_duckdb_json_value(value: object) -> object:
    """Normalize a Python value for DuckDB JSON column insertion.

    DuckDB's JSON column type accepts native Python containers (dict/list).
    This helper keeps containers in native form and coerces non-JSON container
    types (sets/tuples) into JSON-compatible lists.

    Parameters
    ----------
    value
        Value destined for a DuckDB JSON-typed column.

    Returns
    -------
    object
        JSON-compatible Python object suitable for JSON-typed columns.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        decoded = _decode_bytes_payload(value)
        return _coerce_json_container(_coerce_decoded_payload(decoded))
    normalized = _coerce_json_container(value)
    if isinstance(normalized, str):
        parsed = _parse_json_value(normalized)
        if parsed is not None:
            normalized = parsed
    return _coerce_json_container(normalized)


def serialize_str_sequence(items: Sequence[str]) -> list[str]:
    """Normalize a sequence of strings for JSON column storage.

    Parameters
    ----------
    items
        Sequence of strings to normalize.

    Returns
    -------
    list[str]
        JSON-compatible list of strings.
    """
    return list(items)


def deserialize_str_tuple(raw: object | None) -> tuple[str, ...]:
    """Deserialize JSON array to string tuple.

    Parameters
    ----------
    raw
        JSON array as string, list, or None.

    Returns
    -------
    tuple[str, ...]
        Tuple of strings, empty if raw is None or empty.
    """
    if raw in {None, ""}:
        return ()
    items = decode_json_list(raw)
    return tuple(str(x) for x in items)
