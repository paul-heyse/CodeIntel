"""Fingerprinting utilities.

This module provides utilities for creating fingerprints of
objects and data structures.
"""

from __future__ import annotations

import hashlib

from codeintel.core.serialization.stable import stable_stringify


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
    serialized = stable_stringify(data)
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
    parts = [stable_stringify(arg) for arg in args]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode(), usedforsecurity=False).hexdigest()


__all__ = [
    "fingerprint",
    "stable_hash",
]
