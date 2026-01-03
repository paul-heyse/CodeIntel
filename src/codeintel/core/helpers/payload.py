"""Msgspec payload helpers for binary contract columns.

Deprecated: use ``codeintel.core.serialization.payload``.
"""

from __future__ import annotations

from codeintel.core.serialization.payload import PayloadValue, decode_payload, encode_payload

__all__ = [
    "PayloadValue",
    "decode_payload",
    "encode_payload",
]
