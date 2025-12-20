"""DuckDB type aliases and error classes for storage-only usage."""

from __future__ import annotations

import duckdb

DuckDBConnection = duckdb.DuckDBPyConnection
DuckDBRelation = duckdb.DuckDBPyRelation
DuckDBError = duckdb.Error
DuckDBCatalogException = duckdb.CatalogException
DuckDBConnectionException = duckdb.ConnectionException
DuckDBDatabaseError = duckdb.DatabaseError
DuckDBInvalidInputException = duckdb.InvalidInputException
DuckDBProgrammingError = duckdb.ProgrammingError
DuckDBBinderException = duckdb.BinderException

__all__ = [
    "DuckDBBinderException",
    "DuckDBCatalogException",
    "DuckDBConnection",
    "DuckDBConnectionException",
    "DuckDBDatabaseError",
    "DuckDBError",
    "DuckDBInvalidInputException",
    "DuckDBProgrammingError",
    "DuckDBRelation",
]
