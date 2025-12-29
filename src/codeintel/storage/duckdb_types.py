"""DuckDB type aliases and error classes for storage-only usage."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import duckdb

DuckDBConnection = duckdb.DuckDBPyConnection
DuckDBRelation = duckdb.DuckDBPyRelation
DuckDBError = duckdb.Error
Expression = duckdb.Expression
type ExpressionFactory = Callable[..., Expression]
type ColumnExpressionFactory = Callable[[str], Expression]
type ConstantExpressionFactory = Callable[[object], Expression]
type FunctionExpressionFactory = Callable[..., Expression]
ColumnExpression = cast("ColumnExpressionFactory", duckdb.ColumnExpression)
ConstantExpression = cast("ConstantExpressionFactory", duckdb.ConstantExpression)
FunctionExpression = cast("FunctionExpressionFactory", duckdb.FunctionExpression)
DuckDBCatalogException = duckdb.CatalogException
DuckDBConnectionException = duckdb.ConnectionException
DuckDBDatabaseError = duckdb.DatabaseError
DuckDBInvalidInputException = duckdb.InvalidInputException
DuckDBProgrammingError = duckdb.ProgrammingError
DuckDBBinderException = duckdb.BinderException

__all__ = [
    "ColumnExpression",
    "ColumnExpressionFactory",
    "ConstantExpression",
    "ConstantExpressionFactory",
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
    "ExpressionFactory",
    "FunctionExpression",
    "FunctionExpressionFactory",
]
