"""Msgspec-backed serialization helpers."""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import TypeGuard, Union, get_args, get_origin

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
    fields = msgspec_structs.fields(target_type)
    allowed = {field.encode_name: field.type for field in fields}
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        normalized[key] = strip_unknown_fields(item, allowed[key])
    return normalized


def _is_struct_type(target_type: object) -> TypeGuard[type[msgspec.Struct]]:
    return isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)


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
    "strip_unknown_fields",
    "to_builtins",
]
