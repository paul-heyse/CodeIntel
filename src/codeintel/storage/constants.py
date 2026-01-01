"""Shared constants for the storage layer.

This module centralizes constants used across multiple storage modules
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE, DUCKDB_DIALECT, SCHEMAS

__all__ = [
    "DEFAULT_ARROW_BATCH_SIZE",
    "DUCKDB_DIALECT",
    "META_CATALOG_NAME",
    "META_DB_FILENAME",
    "SCHEMAS",
]

META_CATALOG_NAME = "meta"
"""Catalog name for the attached meta database."""

META_DB_FILENAME = "meta.duckdb"
"""Default filename for the meta database."""
