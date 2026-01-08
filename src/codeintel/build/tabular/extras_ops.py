"""Extras normalization helpers for graph metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping

from codeintel.core.serialization.payload import decode_payload

ExtrasMapping = Mapping[str, object] | Mapping[object, object]


def extras_kv_value(value: object) -> str:
    """Coerce an extras value into a string.

    Parameters
    ----------
    value
        Value to serialize into a string representation.

    Returns
    -------
    str
        String-encoded extras value.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def extras_kv_from_mapping(values: ExtrasMapping | None) -> dict[str, str] | None:
    """Build an extras_kv map from a mapping payload.

    Parameters
    ----------
    values
        Mapping containing extras metadata.

    Returns
    -------
    dict[str, str] | None
        Stringified extras mapping, or None when empty.
    """
    if not values:
        return None
    extras: dict[str, str] = {}
    for key, item in values.items():
        if item is None:
            continue
        extras[str(key)] = extras_kv_value(item)
    return extras or None


def extras_kv_from_payload(value: object) -> dict[str, str] | None:
    """Decode a payload value into an extras_kv mapping.

    Parameters
    ----------
    value
        Payload bytes or decoded extras payload.

    Returns
    -------
    dict[str, str] | None
        Extras mapping derived from the payload.
    """
    decoded = decode_payload(value)
    if decoded is None:
        return None
    if isinstance(decoded, Mapping):
        return extras_kv_from_mapping(decoded)
    return {"value": extras_kv_value(decoded)}


__all__ = ["extras_kv_from_mapping", "extras_kv_from_payload", "extras_kv_value"]
