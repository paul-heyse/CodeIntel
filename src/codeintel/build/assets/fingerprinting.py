"""Asset fingerprinting helpers for Phase 4.

This module intentionally lives in the build layer: it computes deterministic
fingerprints for datasets and artifacts without depending on storage accessors.
"""

from __future__ import annotations

import hashlib

from codeintel.config.datasets.schemas import TABLE_SCHEMAS


def _canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def compute_table_schema_hash(table_key: str) -> str | None:
    """Return a deterministic schema hash for a known dataset table_key.

    Parameters
    ----------
    table_key
        Fully-qualified dataset table key (e.g., "analytics.function_metrics").

    Returns
    -------
    str | None
        SHA256 hex digest of (column_name:type) pairs, or None if table_key
        is not registered or has no schema (e.g., view).
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return None
    if schema.columns is None:
        return None
    parts = [f"{column.name}:{_canonical_type(column.type)}" for column in schema.columns]
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_fast_version_hash(*parts: object) -> str:
    """Compute a fast, stable version hash from a tuple of stable components.

    Notes
    -----
    This intentionally produces a short hash for ergonomics. It is content
    addressed but not collision-proof; Phase 4 can upgrade to a stronger policy.

    Returns
    -------
    str
        Hex digest truncated to 16 characters.
    """
    normalized = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "compute_fast_version_hash",
    "compute_table_schema_hash",
]
