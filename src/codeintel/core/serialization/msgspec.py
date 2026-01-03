"""Msgspec-backed serialization helpers."""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Literal, TypeGuard, Union, cast, get_args, get_origin

import msgspec
import msgspec.structs as msgspec_structs


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

    Parameters
    ----------
    payload
        Payload to serialize.
    indent
        Optional indentation for pretty formatting.
    newline
        Whether to append a trailing newline.

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

    Parameters
    ----------
    payload
        Payload to serialize.
    indent
        Optional indentation for pretty formatting.
    newline
        Whether to append a trailing newline.

    Returns
    -------
    str
        JSON-encoded payload text.
    """
    return encode_json_bytes(payload, indent=indent, newline=newline).decode("utf-8")


def encode_json_lines(payloads: Iterable[object]) -> bytes:
    """Encode an iterable payload as JSON Lines bytes.

    Parameters
    ----------
    payloads
        Iterable of payloads to encode.

    Returns
    -------
    bytes
        JSON Lines-encoded payload bytes.
    """
    return JSON_ENCODER.encode_lines(payloads)


def encode_json_line_text(payload: object) -> str:
    """Encode a single payload as JSON Lines text.

    Parameters
    ----------
    payload
        Payload to serialize.

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

    Parameters
    ----------
    payload
        JSON-encoded bytes payload.
    payload_type
        Target type for decoding.

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

    Parameters
    ----------
    payload
        JSON-encoded text payload.
    payload_type
        Target type for decoding.

    Returns
    -------
    T
        Decoded payload instance.
    """
    return decode_json_bytes(payload.encode("utf-8"), payload_type=payload_type)


def decode_boundary_payload[T](
    payload: object,
    *,
    payload_type: type[T],
) -> T:
    """Decode a boundary payload into a typed object.

    Parameters
    ----------
    payload
        Boundary payload (bytes, text, or JSON-like builtins).
    payload_type
        Target type for decoding.

    Returns
    -------
    T
        Decoded payload instance.

    Raises
    ------
    TypeError
        Raised when the payload type is unsupported.
    """
    if isinstance(payload_type, type) and isinstance(payload, payload_type):
        return cast("T", payload)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return _decode_boundary_bytes(bytes(payload), payload_type)
    if isinstance(payload, str):
        return _decode_boundary_text(payload, payload_type)
    if isinstance(payload, (Mapping, list, tuple)):
        return _convert_boundary_payload(payload, payload_type)
    msg = f"Unsupported payload type: {type(payload).__name__}"
    raise TypeError(msg)


def encode_boundary_payload(
    payload: object,
    *,
    payload_format: Literal["json", "msgpack"] = "json",
    indent: int | None = 2,
    newline: bool = False,
) -> bytes:
    """Encode a payload for boundary transport.

    Parameters
    ----------
    payload
        Payload to encode.
    payload_format
        Encoding format ("json" or "msgpack").
    indent
        Optional indentation for JSON formatting.
    newline
        Whether to append a trailing newline for JSON output.

    Returns
    -------
    bytes
        Encoded payload bytes.

    Raises
    ------
    ValueError
        Raised when an unsupported format is requested.
    """
    if payload_format == "msgpack":
        return msgspec.msgpack.encode(payload)
    if payload_format == "json":
        return encode_json_bytes(payload, indent=indent, newline=newline)
    msg = f"Unsupported boundary format: {payload_format}"
    raise ValueError(msg)


def to_builtins(payload: object) -> object:
    """Convert supported objects into JSON-serializable builtins.

    Parameters
    ----------
    payload
        Payload to convert.

    Returns
    -------
    object
        Builtin representation suitable for msgspec JSON encoding.
    """
    return msgspec.to_builtins(payload, enc_hook=_encode_hook)


def schema_for(payload_type: type[object]) -> dict[str, object]:
    """Generate JSON Schema for a msgspec-compatible type.

    Parameters
    ----------
    payload_type
        Type to generate schema for.

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

    Parameters
    ----------
    types
        Types to include in schema generation.

    Returns
    -------
    tuple[tuple[dict[str, object], ...], dict[str, object]]
        Tuple of schema entries and $defs dictionary.
    """
    return msgspec.json.schema_components(types)


def strip_unknown_fields(value: object, target_type: object) -> object:
    """Strip unknown fields from JSON-like payloads for msgspec decoding.

    Parameters
    ----------
    value
        Parsed JSON-like payload.
    target_type
        Target type for msgspec conversion.

    Returns
    -------
    object
        Payload with unknown fields removed where applicable.
    """
    origin = get_origin(target_type)
    if origin is None:
        if _is_struct_type(target_type):
            return _strip_struct_fields(value, target_type)
        return value
    if origin in {list, tuple, set, frozenset}:
        return _strip_collection_fields(value, target_type, origin)
    if origin in {dict, Mapping}:
        return _strip_mapping_fields(value, target_type)
    if origin in {Union, types.UnionType}:
        return _strip_union_fields(value, target_type)
    return value


def _strip_collection_fields(value: object, target_type: object, origin: object) -> object:
    args = get_args(target_type)
    item_type = args[0] if args else object
    if isinstance(value, list):
        items = [strip_unknown_fields(item, item_type) for item in value]
        if origin is tuple:
            return tuple(items)
        if origin is set:
            return set(items)
        if origin is frozenset:
            return frozenset(items)
        return items
    if origin is tuple and isinstance(value, tuple):
        return tuple(strip_unknown_fields(item, item_type) for item in value)
    return value


def _strip_mapping_fields(value: object, target_type: object) -> object:
    if not isinstance(value, dict):
        return value
    args = get_args(target_type)
    key_type = args[0] if args else object
    value_type = args[1] if len(args) > 1 else object
    normalized: dict[object, object] = {}
    for key, item in value.items():
        normalized_key = str(key) if key_type is str else key
        normalized[normalized_key] = strip_unknown_fields(item, value_type)
    return normalized


def _strip_union_fields(value: object, target_type: object) -> object:
    if value is None:
        return None
    for arg in (arg for arg in get_args(target_type) if arg is not type(None)):
        if _is_struct_type(arg) and isinstance(value, dict):
            return strip_unknown_fields(value, arg)
        origin = get_origin(arg)
        if origin is not None:
            return strip_unknown_fields(value, arg)
    return value


def _strip_struct_fields(value: object, target_type: type[msgspec.Struct]) -> object:
    if not isinstance(value, dict):
        return value
    fields = msgspec_structs.fields(cast("msgspec.Struct", target_type))
    allowed = {field.encode_name: field.type for field in fields}
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        normalized[key] = strip_unknown_fields(item, allowed[key])
    return normalized


def _decode_boundary_bytes[T](payload: bytes, payload_type: type[T]) -> T:
    try:
        return msgspec.msgpack.decode(payload, type=payload_type)
    except (msgspec.DecodeError, msgspec.ValidationError):
        pass
    try:
        return msgspec.json.decode(payload, type=payload_type, strict=True)
    except msgspec.ValidationError as exc:
        builtins = msgspec.json.decode(payload)
        sanitized = strip_unknown_fields(builtins, payload_type)
        try:
            return msgspec.convert(sanitized, type=payload_type, strict=True)
        except msgspec.ValidationError as fallback_exc:
            raise fallback_exc from exc


def _decode_boundary_text[T](payload: str, payload_type: type[T]) -> T:
    return _decode_boundary_bytes(payload.encode("utf-8"), payload_type)


def _convert_boundary_payload[T](payload: object, payload_type: type[T]) -> T:
    sanitized = strip_unknown_fields(payload, payload_type)
    return msgspec.convert(sanitized, type=payload_type, strict=True)


def _is_struct_type(target_type: object) -> TypeGuard[type[msgspec.Struct]]:
    return isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)


__all__ = [
    "JSON_DECODER",
    "JSON_ENCODER",
    "decode_boundary_payload",
    "decode_json_bytes",
    "decode_json_text",
    "encode_boundary_payload",
    "encode_json_bytes",
    "encode_json_line_text",
    "encode_json_lines",
    "encode_json_text",
    "schema_components",
    "schema_for",
    "strip_unknown_fields",
    "to_builtins",
]
