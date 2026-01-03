"""Serialization utilities for CodeIntel.

This module provides the canonical value conversion helpers used across
the codebase for JSON-compatible payloads.
"""

from __future__ import annotations

from codeintel.core.serialization.converters import (
    deserialize_value,
    serialize_dataclass_to_dict,
    serialize_value,
)
from codeintel.core.serialization.json import (
    decode_json,
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    normalize_duckdb_json_value,
    serialize_str_sequence,
)
from codeintel.core.serialization.payload import PayloadValue, decode_payload, encode_payload

__all__ = [
    "PayloadValue",
    "decode_json",
    "decode_json_dict",
    "decode_json_list",
    "decode_payload",
    "deserialize_str_tuple",
    "deserialize_value",
    "encode_json_compact",
    "encode_payload",
    "normalize_duckdb_json_value",
    "serialize_dataclass_to_dict",
    "serialize_str_sequence",
    "serialize_value",
]
