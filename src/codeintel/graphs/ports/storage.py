"""Storage port interface for database operations.

This module defines the StoragePort protocol that abstracts all database
interactions, allowing pure computation to be decoupled from DuckDB specifics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class QueryResult:
    """Result of a database query operation.

    Attributes
    ----------
    rows
        Sequence of row tuples returned by the query.
    row_count
        Number of rows returned.
    """

    rows: tuple[tuple[object, ...], ...]
    row_count: int

    @classmethod
    def empty(cls) -> QueryResult:
        """Create an empty query result.

        Returns
        -------
        QueryResult
            Result with no rows.
        """
        return cls(rows=(), row_count=0)

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[object, ...]]) -> QueryResult:
        """Create a query result from a sequence of rows.

        Parameters
        ----------
        rows
            Row tuples to include in the result.

        Returns
        -------
        QueryResult
            Result containing the provided rows.
        """
        row_tuple = tuple(rows)
        return cls(rows=row_tuple, row_count=len(row_tuple))


@dataclass(frozen=True)
class BatchResult:
    """Result of a batch insert/update operation.

    Attributes
    ----------
    rows_affected
        Number of rows inserted or updated.
    table
        Target table name.
    success
        Whether the operation succeeded.
    error
        Error message if operation failed.
    """

    rows_affected: int
    table: str
    success: bool = True
    error: str | None = None

    @classmethod
    def ok(cls, table: str, rows_affected: int) -> BatchResult:
        """Create a successful batch result.

        Parameters
        ----------
        table
            Target table name.
        rows_affected
            Number of rows affected.

        Returns
        -------
        BatchResult
            Successful result.
        """
        return cls(rows_affected=rows_affected, table=table, success=True)

    @classmethod
    def fail(cls, table: str, error: str) -> BatchResult:
        """Create a failed batch result.

        Parameters
        ----------
        table
            Target table name.
        error
            Error message describing the failure.

        Returns
        -------
        BatchResult
            Failed result with error message.
        """
        return cls(rows_affected=0, table=table, success=False, error=error)


@runtime_checkable
class StoragePort(Protocol):
    """Protocol for database storage operations.

    Implementations provide access to query execution, batch operations,
    and source file reading without exposing database-specific details.
    """

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a SQL query and return results.

        Parameters
        ----------
        sql
            SQL query string with optional parameter placeholders.
        params
            Parameter values to bind to placeholders.

        Returns
        -------
        QueryResult
            Query results containing rows and metadata.
        """
        ...

    def execute_mutation(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> int:
        """Execute a SQL mutation (INSERT/UPDATE/DELETE).

        Parameters
        ----------
        sql
            SQL mutation statement.
        params
            Parameter values to bind.

        Returns
        -------
        int
            Number of rows affected.
        """
        ...

    def run_batch(
        self,
        table: str,
        rows: Sequence[tuple[object, ...]],
        *,
        delete_params: Sequence[object] | None = None,
        scope: str | None = None,
    ) -> BatchResult:
        """Insert a batch of rows into a table.

        Parameters
        ----------
        table
            Target table name (e.g., "graph.call_graph_edges").
        rows
            Sequence of row tuples to insert.
        delete_params
            Optional parameters for pre-delete operation.
        scope
            Scope identifier for logging/tracking.

        Returns
        -------
        BatchResult
            Result of the batch operation.
        """
        ...

    def read_source(self, rel_path: str) -> str | None:
        """Read source file contents.

        Parameters
        ----------
        rel_path
            Relative path to the source file.

        Returns
        -------
        str | None
            File contents if readable, None otherwise.
        """
        ...

    @property
    def repo_root(self) -> Path:
        """Repository root path.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        ...


__all__ = [
    "BatchResult",
    "QueryResult",
    "StoragePort",
]
