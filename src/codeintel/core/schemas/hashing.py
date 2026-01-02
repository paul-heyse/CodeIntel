"""Canonical schema hashing utilities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from codeintel.core.hashing.fingerprint import fingerprint

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider

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


def schema_digest(schema: TableSchema) -> str:
    """Return a deterministic schema digest for a TableSchema.

    The digest is a fingerprint over the full TableSchema JSON payload.

    Parameters
    ----------
    schema
        Table schema to digest.

    Returns
    -------
    str
        Hex-encoded SHA-256 fingerprint.
    """
    return fingerprint(schema.to_json_obj())


def compute_table_schema_hash(
    table_key: str,
    *,
    schema_provider: SchemaProvider,
) -> str | None:
    """Return a deterministic schema hash for a known dataset table key.

    Parameters
    ----------
    table_key
        Fully-qualified dataset table key (e.g., "core.modules").
    schema_provider
        Schema provider used to resolve the table schema.

    Returns
    -------
    str | None
        SHA256 hex digest of (column_name:type) pairs, or None if table_key
        is not registered or has no schema (e.g., view).
    """
    schema = schema_provider.get_table_schema(table_key)
    if schema is None:
        return None
    return schema_hash(schema)


__all__ = [
    "canonical_type",
    "compute_table_schema_hash",
    "schema_digest",
    "schema_hash",
]
