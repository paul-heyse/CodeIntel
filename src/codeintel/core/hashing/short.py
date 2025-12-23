"""Short-form hashing helpers for stable identifiers."""

from __future__ import annotations

import hashlib

__all__ = ["sha1_short", "sha256_short", "short_hash"]


def _hash_bytes(algorithm: str, data: bytes, *, used_for_security: bool) -> str:
    try:
        hasher = hashlib.new(algorithm, data, usedforsecurity=used_for_security)
    except TypeError:
        hasher = hashlib.new(algorithm, data)
    return hasher.hexdigest()


def short_hash(
    text: str,
    *,
    algorithm: str = "sha256",
    length: int = 16,
    used_for_security: bool = False,
) -> str:
    """Return a short hex digest for a string payload.

    Parameters
    ----------
    text
        Input string to hash.
    algorithm
        Hash algorithm name (e.g., "sha1", "sha256").
    length
        Prefix length for the digest.
    used_for_security
        Whether the hash is used for security-sensitive purposes.

    Returns
    -------
    str
        Hex digest prefix of the requested length.
    """
    digest = _hash_bytes(
        algorithm,
        text.encode("utf-8"),
        used_for_security=used_for_security,
    )
    if length <= 0:
        return ""
    return digest[:length]


def sha1_short(text: str, *, length: int = 16, used_for_security: bool = False) -> str:
    """Return a short SHA-1 hex digest for a string payload.

    Returns
    -------
    str
        Hex digest prefix of the requested length.
    """
    return short_hash(
        text,
        algorithm="sha1",
        length=length,
        used_for_security=used_for_security,
    )


def sha256_short(text: str, *, length: int = 16, used_for_security: bool = False) -> str:
    """Return a short SHA-256 hex digest for a string payload.

    Returns
    -------
    str
        Hex digest prefix of the requested length.
    """
    return short_hash(
        text,
        algorithm="sha256",
        length=length,
        used_for_security=used_for_security,
    )
