"""Cache key generation utilities.

This module provides utilities for generating consistent cache keys
from various input types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from codeintel.core.serialization.stable import stable_stringify


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
            parts.append(f"{k}={stable_stringify(v)}")

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
            self._parts.append(f"{k}={stable_stringify(v)}")
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
