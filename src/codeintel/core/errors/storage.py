"""Storage and query error types.

This module provides error types for database operations, including
query failures, missing tables/columns, and schema issues.
"""

from __future__ import annotations

from codeintel.core.errors.base import CodeIntelStorageError
from codeintel.core.errors.taxonomy import (
    COLUMN_NOT_FOUND,
    CONNECTION_FAILED,
    QUERY_FAILED,
    TABLE_NOT_FOUND,
)


class QueryError(Exception):
    """Base class for query execution errors.

    Attributes
    ----------
    table
        The table involved in the failed query.
    message
        Description of the error.
    """

    def __init__(self, table: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        table
            The table involved in the failed query.
        message
            Description of the error.
        """
        self.table = table
        super().__init__(f"Query error on {table}: {message}")


class TableNotFoundError(QueryError):
    """Table does not exist in the database."""


class ColumnNotFoundError(QueryError):
    """Column does not exist in the table.

    Attributes
    ----------
    column
        The missing column name.
    """

    def __init__(self, table: str, column: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        table
            The table being queried.
        column
            The missing column name.
        """
        self.column = column
        super().__init__(table, f"column '{column}' not found")


class StorageError(CodeIntelStorageError):
    """Storage error with structured Problem Details support."""

    def __init__(
        self,
        message: str,
        *,
        table: str | None = None,
        query: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize storage error.

        Parameters
        ----------
        message
            Error message.
        table
            Table involved in the error.
        query
            SQL query that failed (truncated to 200 chars).
        cause
            Underlying exception.
        """
        super().__init__(
            error_code=QUERY_FAILED,
            message=message,
            table=table,
            query=query,
            cause=cause,
        )


class StorageConnectionError(CodeIntelStorageError):
    """Storage connection error with structured Problem Details support."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        """Initialize storage connection error.

        Parameters
        ----------
        message
            Error message.
        cause
            Underlying exception.
        """
        super().__init__(
            error_code=CONNECTION_FAILED,
            message=message,
            cause=cause,
        )


class StorageQueryError(CodeIntelStorageError):
    """Structured query error with Problem Details support.

    Use this when you need structured error handling with RFC 9457
    Problem Details support.
    """

    def __init__(
        self,
        message: str,
        *,
        table: str | None = None,
        query: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize storage query error.

        Parameters
        ----------
        message
            Error message.
        table
            Table involved in the error.
        query
            SQL query that failed (truncated to 200 chars).
        cause
            Underlying exception.
        """
        super().__init__(
            error_code=QUERY_FAILED,
            message=message,
            table=table,
            query=query,
            cause=cause,
        )


class StorageTableNotFoundError(CodeIntelStorageError):
    """Structured table not found error with Problem Details support."""

    def __init__(
        self,
        table: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        """Initialize table not found error.

        Parameters
        ----------
        table
            The missing table name.
        cause
            Underlying exception.
        """
        super().__init__(
            error_code=TABLE_NOT_FOUND,
            message=f"Table not found: {table}",
            table=table,
            cause=cause,
        )


class StorageColumnNotFoundError(CodeIntelStorageError):
    """Structured column not found error with Problem Details support."""

    def __init__(
        self,
        table: str,
        column: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        """Initialize column not found error.

        Parameters
        ----------
        table
            The table being queried.
        column
            The missing column name.
        cause
            Underlying exception.
        """
        super().__init__(
            error_code=COLUMN_NOT_FOUND,
            message=f"Column '{column}' not found in table '{table}'",
            table=table,
            cause=cause,
        )
        self.column = column


__all__ = [
    "ColumnNotFoundError",
    "QueryError",
    "StorageColumnNotFoundError",
    "StorageConnectionError",
    "StorageError",
    "StorageQueryError",
    "StorageTableNotFoundError",
    "TableNotFoundError",
]
