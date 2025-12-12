"""Storage error types and exception tuples.

This module provides common error types used across the storage layer.
It has NO internal dependencies to prevent circular imports.

Note
----
This module intentionally has no imports from other codeintel packages.
Only standard library and third-party packages are imported here.
"""

from __future__ import annotations

from duckdb import Error as DuckDBError

DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)

__all__ = [
    "DUCKDB_ERRORS",
    "DuckDBError",
]
