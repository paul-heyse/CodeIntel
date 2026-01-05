"""Msgspec payload helpers for binary contract columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import msgspec

PayloadValue = Mapping[str, object] | Sequence[object] | str | int | float | bool | None

_INT64_MIN = -(2**63)
_UINT64_MAX = 2**64 - 1


def _sanitize_payload(value: object) -> object:
    if value is None:
        result: object = None
    elif isinstance(value, bool):
        result = value
    elif isinstance(value, int):
        result = str(value) if value < _INT64_MIN or value > _UINT64_MAX else value
    elif isinstance(value, (str, float)):
        result = value
    elif isinstance(value, Mapping):
        sanitized: dict[object, object] = {}
        for key, item in value.items():
            safe_key = _sanitize_payload(key)
            if isinstance(safe_key, (bytes, bytearray, memoryview)):
                safe_key = str(bytes(safe_key))
            sanitized[safe_key] = _sanitize_payload(item)
        result = sanitized
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        result = [_sanitize_payload(item) for item in value]
    else:
        result = str(value)
    return result


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
    sanitized = _sanitize_payload(value)
    return msgspec.msgpack.encode(sanitized)


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
