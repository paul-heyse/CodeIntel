"""Msgspec-backed serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import msgspec


def _encode_hook(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    msg = f"Unsupported type: {type(value).__name__}"
    raise TypeError(msg)


JSON_ENCODER = msgspec.json.Encoder(
    order="deterministic",
    enc_hook=_encode_hook,
)
JSON_DECODER = msgspec.json.Decoder(strict=True)


def encode_json_bytes(
    payload: object,
    *,
    indent: int | None = 2,
    newline: bool = False,
) -> bytes:
    """Encode a payload as JSON bytes with deterministic ordering.

    Returns
    -------
    bytes
        JSON-encoded payload bytes.
    """
    encoded = JSON_ENCODER.encode(payload)
    if indent is not None:
        encoded = msgspec.json.format(encoded, indent=indent)
    if newline:
        encoded += b"\n"
    return encoded


def encode_json_text(
    payload: object,
    *,
    indent: int | None = 2,
    newline: bool = False,
) -> str:
    """Encode a payload as JSON text with deterministic ordering.

    Returns
    -------
    str
        JSON-encoded payload text.
    """
    return encode_json_bytes(payload, indent=indent, newline=newline).decode("utf-8")


def encode_json_lines(payloads: Iterable[object]) -> bytes:
    """Encode an iterable payload as JSON Lines bytes.

    Returns
    -------
    bytes
        JSON Lines-encoded payload bytes.
    """
    return JSON_ENCODER.encode_lines(payloads)


def encode_json_line_text(payload: object) -> str:
    """Encode a single payload as JSON Lines text.

    Returns
    -------
    str
        JSON-encoded line with trailing newline.
    """
    encoded = JSON_ENCODER.encode(payload).decode("utf-8")
    return f"{encoded}\n"


def decode_json_bytes[T](
    payload: bytes,
    *,
    payload_type: type[T],
) -> T:
    """Decode JSON bytes into a typed payload.

    Returns
    -------
    T
        Decoded payload instance.
    """
    return msgspec.json.decode(payload, type=payload_type, strict=True)


def decode_json_text[T](
    payload: str,
    *,
    payload_type: type[T],
) -> T:
    """Decode JSON text into a typed payload.

    Returns
    -------
    T
        Decoded payload instance.
    """
    return decode_json_bytes(payload.encode("utf-8"), payload_type=payload_type)


def to_builtins(payload: object) -> object:
    """Convert supported objects into JSON-serializable builtins.

    Returns
    -------
    object
        Builtin representation suitable for msgspec JSON encoding.
    """
    return msgspec.to_builtins(payload, enc_hook=_encode_hook)


def schema_for(payload_type: type[object]) -> dict[str, object]:
    """Generate JSON Schema for a msgspec-compatible type.

    Returns
    -------
    dict[str, object]
        JSON Schema for the provided type.
    """
    return msgspec.json.schema(payload_type)


def schema_components(
    types: Iterable[object],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Generate JSON Schema components for a collection of types.

    Returns
    -------
    tuple[tuple[dict[str, object], ...], dict[str, object]]
        Tuple of schema entries and $defs dictionary.
    """
    return msgspec.json.schema_components(types)


__all__ = [
    "JSON_DECODER",
    "JSON_ENCODER",
    "decode_json_bytes",
    "decode_json_text",
    "encode_json_bytes",
    "encode_json_line_text",
    "encode_json_lines",
    "encode_json_text",
    "schema_components",
    "schema_for",
    "to_builtins",
]
