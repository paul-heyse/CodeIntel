"""Storage layer exceptions and error types.

This module defines exception types for the storage layer and re-exports
DuckDB error types for consistent error handling. It provides a clean
abstraction over database-specific errors.

Note
----
This module intentionally has minimal imports from other codeintel packages
to prevent circular imports. Only standard library and third-party packages
are imported here.
"""

from __future__ import annotations

from duckdb import Error as DuckDBError

# Re-export DuckDB errors for catch blocks
DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)


class StorageError(Exception):
    """Base exception for storage layer errors.

    This exception wraps database-specific errors (like duckdb.Error)
    to provide a clean abstraction boundary. Code outside the storage
    layer should catch this instead of database-specific exceptions.
    """


class StorageConnectionError(StorageError):
    """Error establishing or maintaining a database connection."""


class SchemaError(StorageError):
    """Error with database schema (tables, views, macros)."""


class QueryError(StorageError):
    """Error executing a database query."""


__all__ = [
    "DUCKDB_ERRORS",
    "DuckDBError",
    "QueryError",
    "SchemaError",
    "StorageConnectionError",
    "StorageError",
]
