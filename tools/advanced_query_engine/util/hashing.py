"""Stable hashing helpers for IDs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_hex_digest(parts: Iterable[object], *, n: int = 16) -> str:
    """Return a stable SHA-256 digest for string parts.

    Parameters
    ----------
    parts:
        String parts to combine for the digest.
    n:
        Number of hex characters to return.

    Returns
    -------
    str
        Truncated hex digest.
    """
    digest = hashlib.sha256()
    sep = "\x1f"
    payload = sep.join("" if part is None else str(part) for part in parts)
    digest.update(payload.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()[:n]


__all__ = ["stable_hex_digest"]
