"""Short-form hashing helpers for stable identifiers."""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array

__all__ = ["sha1_short", "sha256_short", "short_hash"]

_ARROW_HASH_HEX_LEN = 16


def _hash_kernel_hex(text: str) -> str | None:
    try:
        pc.get_function("hash")
    except (AttributeError, KeyError):
        return None
    values = pa.array([text], type=pa.string())
    hashed = require_array(call_compute("hash", [values]), name="hash")
    scalar = hashed[0]
    value = scalar.as_py() if isinstance(scalar, pa.Scalar) else scalar
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        value += 1 << 64
    return f"{value:016x}"


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
    if not used_for_security and 0 < length <= _ARROW_HASH_HEX_LEN:
        kernel_hex = _hash_kernel_hex(text)
        if kernel_hex is not None:
            return kernel_hex[:length]
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
