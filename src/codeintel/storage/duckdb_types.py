"""DuckDB type aliases and error classes for storage-only usage."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import duckdb

from codeintel.core.schemas.type_mappings import normalize_engine_column_type

if TYPE_CHECKING:
    from duckdb.typing import DuckDBPyType

    from codeintel.core.schemas.primitives import ColumnType

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


def duckdb_type_for_column_type(column_type: ColumnType | None) -> DuckDBPyType | None:
    """Return the DuckDB type for a ColumnType string.

    Parameters
    ----------
    column_type
        Column type string to convert.

    Returns
    -------
    duckdb.typing.DuckDBPyType | None
        DuckDB type when available, otherwise None.
    """
    normalized = normalize_engine_column_type(column_type)
    if normalized is None:
        return None
    try:
        return duckdb.sqltype(normalized)
    except (TypeError, ValueError):
        return None


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
