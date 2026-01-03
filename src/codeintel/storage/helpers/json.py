"""JSON serialization helpers for DuckDB column values (core re-export)."""

from __future__ import annotations

from codeintel.core.serialization.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    normalize_duckdb_json_value,
    serialize_str_sequence,
)

__all__ = [
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "deserialize_str_tuple",
    "encode_json_compact",
    "normalize_duckdb_json_value",
    "serialize_str_sequence",
]
