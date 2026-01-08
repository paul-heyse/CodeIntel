"""Stable ID helpers for graph assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.compute.goid import DECIMAL_38_MAX
from codeintel.build.tabular.compute_helpers import call_compute, require_array
from codeintel.core.serialization.payload import encode_payload

_UINT64_MAX = 2**64 - 1


def _hash_payload(payload: object) -> pa.Array | pa.ChunkedArray:
    try:
        pc.get_function("hash")
    except (AttributeError, KeyError):
        msg = "Arrow hash kernel is unavailable; upgrade pyarrow to enable it."
        raise RuntimeError(msg) from None
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    result = call_compute("hash", [pa.array([serialized])])
    return require_array(result, name="hash")


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

    Raises
    ------
    ValueError
        If the modulus is invalid or the hash result is null.
    """
    if modulus <= 0:
        msg = "stable_int_hash requires a positive modulus"
        raise ValueError(msg)
    if modulus > _UINT64_MAX:
        msg = "stable_int_hash modulus exceeds uint64 range for Arrow hash"
        raise ValueError(msg)
    wrapped = {"digest_size": digest_size, "payload": payload}
    hashed = _hash_payload(wrapped)
    hashed_u64 = pc.cast(hashed, pa.uint64())
    modded = require_array(
        call_compute("mod", [hashed_u64, pa.scalar(modulus, type=pa.uint64())]),
        name="mod",
    )
    value = modded[0].as_py()
    if value is None:
        msg = "Arrow hash produced a null ordinal"
        raise ValueError(msg)
    return int(value)


def stable_decimal_id(payload: object, *, digest_size: int = 16) -> int:
    """Compute a stable DECIMAL(38,0)-safe identifier.

    Returns
    -------
    int
        Stable identifier in the DECIMAL(38,0) range.

    Raises
    ------
    ValueError
        If the hash result is null.
    """
    wrapped = {"digest_size": digest_size, "payload": payload}
    hashed = _hash_payload(wrapped)
    hashed_u64 = pc.cast(hashed, pa.uint64())
    value = hashed_u64[0].as_py()
    if value is None:
        msg = "Arrow hash produced a null identifier"
        raise ValueError(msg)
    int_value = int(value)
    if int_value >= DECIMAL_38_MAX:
        return int_value % DECIMAL_38_MAX
    return int_value


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
