"""Content hashing utilities.

This module provides utilities for hashing content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def content_hash(content: str | bytes, *, algorithm: str = "sha256") -> str:
    """Compute a hash of content.

    Parameters
    ----------
    content
        Content to hash.
    algorithm
        Hash algorithm to use.

    Returns
    -------
    str
        Hexadecimal hash string.

    Examples
    --------
    >>> content_hash("hello world")
    'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    hasher = hashlib.new(algorithm, usedforsecurity=False)
    hasher.update(content)
    return hasher.hexdigest()


def file_hash(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Compute a hash of a file's contents.

    Parameters
    ----------
    path
        Path to the file.
    algorithm
        Hash algorithm to use.

    Returns
    -------
    str
        Hexadecimal hash string.

    Examples
    --------
    >>> file_hash("script.py")
    'abc123...'
    """
    path_obj = Path(path)
    content = path_obj.read_bytes()
    return content_hash(content, algorithm=algorithm)


def content_hash_short(content: str | bytes, *, length: int = 8) -> str:
    """Compute a short hash of content.

    Parameters
    ----------
    content
        Content to hash.
    length
        Length of the returned hash.

    Returns
    -------
    str
        Short hexadecimal hash string.

    Examples
    --------
    >>> content_hash_short("hello world")
    'b94d27b9'
    """
    full_hash = content_hash(content)
    return full_hash[:length]


__all__ = [
    "content_hash",
    "content_hash_short",
    "file_hash",
]
