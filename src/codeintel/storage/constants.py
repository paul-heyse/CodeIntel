"""Shared constants for the storage layer.

This module centralizes constants used across multiple storage modules
to avoid duplication and ensure consistency.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_ARROW_BATCH_SIZE",
    "DUCKDB_DIALECT",
    "META_CATALOG_NAME",
    "META_DB_FILENAME",
    "SCHEMAS",
]

DUCKDB_DIALECT = "duckdb"
"""SQLGlot dialect identifier for DuckDB SQL generation."""

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
"""Database schema names used in the CodeIntel data warehouse."""

META_CATALOG_NAME = "meta"
"""Catalog name for the attached meta database."""

META_DB_FILENAME = "meta.duckdb"
"""Default filename for the meta database."""

DEFAULT_ARROW_BATCH_SIZE = 10_000
"""Default rows per Arrow record batch for streaming exports."""
