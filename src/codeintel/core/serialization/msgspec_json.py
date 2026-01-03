"""Msgspec-backed JSON encoding and decoding utilities."""

from __future__ import annotations

from codeintel.core.serialization.msgspec import (
    JSON_DECODER,
    JSON_ENCODER,
    decode_json_bytes,
    encode_json_bytes,
    encode_json_lines,
    encode_json_text,
)

__all__ = [
    "JSON_DECODER",
    "JSON_ENCODER",
    "decode_json_bytes",
    "encode_json_bytes",
    "encode_json_lines",
    "encode_json_text",
]
