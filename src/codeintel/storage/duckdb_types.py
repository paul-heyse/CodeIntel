"""DuckDB type aliases and error classes for storage-only usage."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import duckdb

DuckDBConnection = duckdb.DuckDBPyConnection
DuckDBRelation = duckdb.DuckDBPyRelation
DuckDBError = duckdb.Error
ColumnExpression = duckdb.ColumnExpression
Expression = duckdb.Expression
ConstantExpression = cast("Callable[[object], Expression]", duckdb.ConstantExpression)
FunctionExpression = cast("Callable[..., Expression]", duckdb.FunctionExpression)
DuckDBCatalogException = duckdb.CatalogException
DuckDBConnectionException = duckdb.ConnectionException
DuckDBDatabaseError = duckdb.DatabaseError
DuckDBInvalidInputException = duckdb.InvalidInputException
DuckDBProgrammingError = duckdb.ProgrammingError
DuckDBBinderException = duckdb.BinderException

__all__ = [
    "ColumnExpression",
    "ConstantExpression",
    "DuckDBBinderException",
    "DuckDBCatalogException",
    "DuckDBConnection",
    "DuckDBConnectionException",
    "DuckDBDatabaseError",
    "DuckDBError",
    "DuckDBInvalidInputException",
    "DuckDBProgrammingError",
    "DuckDBRelation",
    "Expression",
    "FunctionExpression",
]
