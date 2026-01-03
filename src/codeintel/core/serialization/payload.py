"""Msgspec payload helpers for binary contract columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import msgspec

PayloadValue = Mapping[str, object] | Sequence[object] | str | int | float | bool | None


def encode_payload(value: PayloadValue | bytes | bytearray | memoryview | None) -> bytes | None:
    """Encode a payload value to msgpack bytes for storage.

    Parameters
    ----------
    value
        Payload value or bytes-like object.

    Returns
    -------
    bytes | None
        Msgpack-encoded bytes or None when no payload is present.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return msgspec.msgpack.encode(value)


def decode_payload(value: object) -> object | None:
    """Decode a payload value from msgpack bytes when needed.

    Parameters
    ----------
    value
        Stored payload value.

    Returns
    -------
    object | None
        Decoded payload or None if the value is empty.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return msgspec.msgpack.decode(raw)
        except msgspec.DecodeError:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
            try:
                return msgspec.json.decode(text)
            except msgspec.DecodeError:
                return text
    return value


__all__ = [
    "PayloadValue",
    "decode_payload",
    "encode_payload",
]
