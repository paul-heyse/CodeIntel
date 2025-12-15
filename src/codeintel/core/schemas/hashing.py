"""Canonical schema hashing utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

_CANON: dict[str, str] = {
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
    "TIMESTAMP_TZ": "TIMESTAMPTZ",
}


def canonical_type(type_str: str) -> str:
    """Return a canonical type string for stable schema hashing.

    Parameters
    ----------
    type_str
        Raw type string (e.g., from DuckDB DESCRIBE).

    Returns
    -------
    str
        Canonical type string.
    """
    upper = " ".join(type_str.strip().upper().split())
    upper = _CANON.get(upper, upper)
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def schema_hash(schema: TableSchema) -> str:
    """Return a deterministic schema hash for a TableSchema.

    The hash is a stable function of the schema's ordered (name, type) pairs.

    Parameters
    ----------
    schema
        Table schema to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 hash.
    """
    parts = [f"{column.name}:{canonical_type(column.type)}" for column in schema.columns]
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "canonical_type",
    "schema_hash",
]
