"""Storage layer exceptions and error types.

This module re-exports canonical storage errors from
``codeintel.core.errors.storage`` and DuckDB error types for consistent
error handling. It remains as a thin compatibility layer for historical
imports while core owns the canonical error surface.
"""

from __future__ import annotations

from codeintel.core.errors.storage import StorageConnectionError, StorageError
from codeintel.storage.duckdb_types import DuckDBError

# Re-export DuckDB errors for catch blocks
DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)


__all__ = [
    "DUCKDB_ERRORS",
    "DuckDBError",
    "StorageConnectionError",
    "StorageError",
]
