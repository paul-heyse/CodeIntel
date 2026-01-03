"""Fingerprinting utilities.

This module provides utilities for creating fingerprints of
objects and data structures.
"""

from __future__ import annotations

import hashlib

import msgspec

_JSON_ENCODER = msgspec.json.Encoder(order="deterministic", enc_hook=str)


def fingerprint(data: object) -> str:
    """Create a fingerprint of data.

    Creates a deterministic hash of the data structure.

    Parameters
    ----------
    data
        Data to fingerprint.

    Returns
    -------
    str
        Hexadecimal fingerprint.

    Examples
    --------
    >>> fingerprint({"key": "value", "num": 42})
    'abc123...'
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(), usedforsecurity=False).hexdigest()


def stable_hash(*args: object) -> str:
    """Create a stable hash from multiple arguments.

    Creates a deterministic hash from the combined arguments.

    Parameters
    ----------
    *args
        Arguments to hash.

    Returns
    -------
    str
        Hexadecimal hash.

    Examples
    --------
    >>> stable_hash("prefix", 123, {"key": "value"})
    'def456...'
    """
    parts = [_serialize(arg) for arg in args]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode(), usedforsecurity=False).hexdigest()


def _serialize(data: object) -> str:
    """Serialize data for hashing.

    Parameters
    ----------
    data
        Data to serialize.

    Returns
    -------
    str
        Serialized string.
    """
    return _serialize_value(data)


def _serialize_value(data: object) -> str:
    """Serialize a single value.

    Parameters
    ----------
    data
        Value to serialize.

    Returns
    -------
    str
        Serialized string.
    """
    if data is None:
        return "null"
    if isinstance(data, bool):
        return "true" if data else "false"
    if isinstance(data, (int, float, str)):
        return str(data) if not isinstance(data, str) else data
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return _serialize_compound(data)


def _serialize_compound(data: object) -> str:
    """Serialize compound values (list, dict).

    Parameters
    ----------
    data
        Compound value to serialize.

    Returns
    -------
    str
        Serialized string.
    """
    if isinstance(data, (list, tuple)):
        items = [_serialize_value(item) for item in data]
        return f"[{','.join(items)}]"
    if isinstance(data, dict):
        return _JSON_ENCODER.encode(_make_serializable(data)).decode("utf-8")
    return str(data)


def _make_serializable(obj: object) -> object:
    """Make an object JSON-serializable.

    Parameters
    ----------
    obj
        Object to make serializable.

    Returns
    -------
    object
        JSON-serializable object.
    """
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [_make_serializable(item) for item in sorted(obj, key=str)]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "__dict__"):
        return _make_serializable(vars(obj))
    return obj


__all__ = [
    "fingerprint",
    "stable_hash",
]
