"""Storage layer exceptions.

This module defines exception types for the storage layer, providing
a clean abstraction over database-specific errors.
"""

from __future__ import annotations


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
    "QueryError",
    "SchemaError",
    "StorageConnectionError",
    "StorageError",
]
