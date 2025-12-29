"""Compatibility wrapper for Arrow schema helpers."""

from __future__ import annotations

from codeintel.storage.schema.arrow_contracts import (
    arrow_schema_digest,
    arrow_schema_for_table_key,
    arrow_schema_hash,
)

__all__ = [
    "arrow_schema_digest",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
]
