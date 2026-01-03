"""Msgspec-backed JSON encoding and decoding utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import msgspec


def _encode_hook(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
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


def encode_json_lines(payloads: Iterable[object]) -> bytes:
    """Encode an iterable payload as JSON Lines bytes.

    Returns
    -------
    bytes
        JSON Lines-encoded payload bytes.
    """
    return JSON_ENCODER.encode_lines(payloads)


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


__all__ = [
    "JSON_DECODER",
    "JSON_ENCODER",
    "decode_json_bytes",
    "encode_json_bytes",
    "encode_json_lines",
    "encode_json_text",
]
