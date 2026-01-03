"""Arrow schema metadata encoding/decoding utilities."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa

from codeintel.core.serialization.msgspec_json import JSON_DECODER, JSON_ENCODER


def decode_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, object]:
    """Decode Arrow metadata into a JSON-friendly mapping.

    Parameters
    ----------
    metadata
        Arrow metadata mapping of bytes to bytes.

    Returns
    -------
    dict[str, object]
        Decoded metadata values.
    """
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, raw in metadata.items():
        key_str = key.decode("utf-8")
        raw_str = raw.decode("utf-8")
        decoded[key_str] = _decode_metadata_value(raw_str)
    return decoded


def encode_metadata(metadata: Mapping[str, object]) -> dict[bytes, bytes] | None:
    """Encode metadata into Arrow byte mappings.

    Parameters
    ----------
    metadata
        Metadata to encode.

    Returns
    -------
    dict[bytes, bytes] | None
        Encoded metadata or None when empty.
    """
    encoded: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        raw = _encode_metadata_value(value)
        encoded[key.encode("utf-8")] = raw.encode("utf-8")
    return encoded or None


def merge_metadata(
    existing: Mapping[bytes, bytes] | None,
    updates: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> dict[bytes, bytes] | None:
    """Merge metadata updates into an existing Arrow metadata mapping.

    Parameters
    ----------
    existing
        Existing Arrow metadata (bytes to bytes).
    updates
        Metadata updates to apply.
    overwrite
        When True, overwrite existing keys. When False, only fill missing keys.

    Returns
    -------
    dict[bytes, bytes] | None
        Encoded merged metadata.
    """
    if not updates:
        return dict(existing) if existing else None
    merged = dict(decode_metadata(existing))
    for key, value in updates.items():
        if value is None:
            continue
        if not overwrite and key in merged:
            continue
        merged[key] = value
    return encode_metadata(merged)


def merge_field_metadata(
    field: pa.Field,
    updates: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> pa.Field:
    """Return a field with metadata updates applied.

    Parameters
    ----------
    field
        Arrow field to update.
    updates
        Metadata updates to apply.
    overwrite
        When True, overwrite existing keys. When False, only fill missing keys.

    Returns
    -------
    pyarrow.Field
        Updated field with merged metadata.
    """
    merged = merge_metadata(field.metadata, updates, overwrite=overwrite)
    if merged == field.metadata:
        return field
    return field.with_metadata(merged)


def _decode_metadata_value(raw: str) -> object:
    try:
        return JSON_DECODER.decode(raw)
    except ValueError:
        return raw


def _encode_metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return JSON_ENCODER.encode(value).decode("utf-8")


__all__ = [
    "decode_metadata",
    "encode_metadata",
    "merge_field_metadata",
    "merge_metadata",
]
