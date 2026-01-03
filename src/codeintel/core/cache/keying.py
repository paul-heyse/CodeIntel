"""Cache key generation utilities.

This module provides utilities for generating consistent cache keys
from various input types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import msgspec

_JSON_ENCODER = msgspec.json.Encoder(order="deterministic")


def cache_key(*args: object, **kwargs: object) -> str:
    """Generate a cache key from arguments.

    Creates a deterministic string key from the provided arguments.

    Parameters
    ----------
    *args
        Positional arguments to include in key.
    **kwargs
        Keyword arguments to include in key.

    Returns
    -------
    str
        A deterministic string key.

    Examples
    --------
    >>> cache_key("function", "repo/name", 123)
    'function:repo/name:123'
    >>> cache_key("query", filters={"status": "active"})
    'query:filters={"status": "active"}'
    """
    parts = [str(arg) for arg in args]

    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            parts.append(f"{k}={_serialize_value(v)}")

    return ":".join(parts)


def hash_key(*args: object, **kwargs: object) -> str:
    """Generate a hashed cache key from arguments.

    Creates a deterministic hash key, useful when arguments
    may contain long strings or complex objects.

    Parameters
    ----------
    *args
        Positional arguments to include in key.
    **kwargs
        Keyword arguments to include in key.

    Returns
    -------
    str
        A hexadecimal hash string.

    Examples
    --------
    >>> key = hash_key("function", "repo/name", 123)
    >>> len(key)
    32
    """
    base_key = cache_key(*args, **kwargs)
    return hashlib.md5(base_key.encode(), usedforsecurity=False).hexdigest()


def _serialize_value(value: object) -> str:
    """Serialize a value for inclusion in a cache key.

    Parameters
    ----------
    value
        Value to serialize.

    Returns
    -------
    str
        String representation of the value.
    """
    result: str
    if value is None:
        result = "null"
    elif isinstance(value, bool):
        result = "true" if value else "false"
    elif isinstance(value, (int, float, str)):
        result = str(value)
    elif isinstance(value, (list, tuple)):
        items = [_serialize_value(item) for item in value]
        result = f"[{','.join(items)}]"
    elif isinstance(value, dict):
        result = _encode_json(value)
    else:
        result = str(value)
    return result


def _encode_json(value: object) -> str:
    normalized = _normalize_json_value(value)
    return _JSON_ENCODER.encode(normalized).decode("utf-8")


def _normalize_json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        normalized: object = value.decode("utf-8", errors="replace")
    elif isinstance(value, (list, tuple)):
        normalized = [_normalize_json_value(item) for item in value]
    elif isinstance(value, set):
        normalized = [_normalize_json_value(item) for item in sorted(value, key=str)]
    elif isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        normalized = {str(key): _normalize_json_value(val) for key, val in items}
    elif hasattr(value, "__dict__"):
        normalized = _normalize_json_value(vars(value))
    else:
        normalized = str(value)
    return normalized


@dataclass(frozen=True)
class CompositeKey:
    """A composite cache key with multiple components.

    Provides a structured way to create cache keys with
    named components.

    Attributes
    ----------
    namespace
        Key namespace/prefix.
    parts
        Additional key components.

    Examples
    --------
    >>> key = CompositeKey("functions", ("repo/name", "abc123"))
    >>> str(key)
    'functions:repo/name:abc123'
    """

    namespace: str
    parts: tuple[Any, ...] = ()

    def __str__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Key as colon-separated string.
        """
        all_parts = [self.namespace, *[str(p) for p in self.parts]]
        return ":".join(all_parts)

    def with_suffix(self, suffix: str) -> CompositeKey:
        """Create a new key with an additional suffix.

        Parameters
        ----------
        suffix
            Suffix to add.

        Returns
        -------
        CompositeKey
            New key with suffix.
        """
        return CompositeKey(self.namespace, (*self.parts, suffix))

    def to_hash(self) -> str:
        """Return a hashed version of the key.

        Returns
        -------
        str
            MD5 hash of the key.
        """
        return hashlib.md5(str(self).encode(), usedforsecurity=False).hexdigest()


class KeyBuilder:
    """Builder for constructing cache keys.

    Examples
    --------
    >>> key = KeyBuilder("analytics").add("functions").add("repo/name").add(version=1).build()
    """

    def __init__(self, namespace: str) -> None:
        """Initialize the key builder.

        Parameters
        ----------
        namespace
            Key namespace/prefix.
        """
        self._namespace = namespace
        self._parts: list[str] = []

    def add(self, *args: object, **kwargs: object) -> KeyBuilder:
        """Add components to the key.

        Parameters
        ----------
        *args
            Positional components.
        **kwargs
            Named components.

        Returns
        -------
        KeyBuilder
            Self for chaining.
        """
        for arg in args:
            self._parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            self._parts.append(f"{k}={_serialize_value(v)}")
        return self

    def build(self) -> str:
        """Build the final cache key.

        Returns
        -------
        str
            The constructed cache key.
        """
        all_parts = [self._namespace, *self._parts]
        return ":".join(all_parts)

    def build_hash(self) -> str:
        """Build a hashed cache key.

        Returns
        -------
        str
            MD5 hash of the key.
        """
        key = self.build()
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()


__all__ = [
    "CompositeKey",
    "KeyBuilder",
    "cache_key",
    "hash_key",
]
