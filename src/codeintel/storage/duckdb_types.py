"""DuckDB type aliases and error classes for storage usage."""

from __future__ import annotations

from codeintel.core.duckdb_types import (
    ColumnExpression,
    ColumnExpressionFactory,
    ConstantExpression,
    ConstantExpressionFactory,
    DuckDBBinderException,
    DuckDBCatalogException,
    DuckDBConnection,
    DuckDBConnectionException,
    DuckDBDatabaseError,
    DuckDBError,
    DuckDBInvalidInputException,
    DuckDBProgrammingError,
    DuckDBRelation,
    Expression,
    ExpressionFactory,
    FunctionExpression,
    FunctionExpressionFactory,
    duckdb_type_for_column_type,
)

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
    "duckdb_type_for_column_type",
]
