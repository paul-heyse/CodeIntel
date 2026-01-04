"""Stable ID helpers for graph assembly."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from codeintel.build.graphs.compute.goid import DECIMAL_38_MAX
from codeintel.core.serialization.payload import encode_payload


def stable_int_hash(
    payload: object,
    *,
    digest_size: int,
    modulus: int,
) -> int:
    """Compute a stable integer hash for a payload.

    Returns
    -------
    int
        Stable hash modulo the requested modulus.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.blake2b(serialized.encode("utf-8"), digest_size=digest_size).digest()
    return int.from_bytes(digest, "big") % modulus


def stable_decimal_id(payload: object, *, digest_size: int = 16) -> int:
    """Compute a stable DECIMAL(38,0)-safe identifier.

    Returns
    -------
    int
        Stable identifier in the DECIMAL(38,0) range.
    """
    return stable_int_hash(payload, digest_size=digest_size, modulus=DECIMAL_38_MAX)


def payload_bytes(values: Mapping[str, object]) -> bytes:
    """Encode a payload mapping as bytes.

    Returns
    -------
    bytes
        Encoded payload bytes.

    Raises
    ------
    ValueError
        If payload encoding fails.
    """
    encoded = encode_payload(dict(values))
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


__all__ = [
    "payload_bytes",
    "stable_decimal_id",
    "stable_int_hash",
]
