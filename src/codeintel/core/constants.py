"""Shared constants for core/build/serving layers."""

from __future__ import annotations

__all__ = [
    "DEFAULT_ARROW_BATCH_SIZE",
    "DUCKDB_DIALECT",
    "SCHEMAS",
]

DUCKDB_DIALECT = "duckdb"
"""SQLGlot dialect identifier for DuckDB SQL generation."""

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
"""Database schema names used in the CodeIntel data warehouse."""

DEFAULT_ARROW_BATCH_SIZE = 10_000
"""Default rows per Arrow record batch for streaming exports."""
