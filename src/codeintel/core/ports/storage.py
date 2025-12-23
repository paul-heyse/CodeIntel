"""Unified storage port types and protocols.

This module provides canonical storage data types for use across analytics,
graphs, and ingestion packages. It unifies the previously separate storage
port implementations into a single source of truth.

Types
-----
QueryResult
    Unified query result for all storage operations.
BatchResult
    Unified batch operation result.
StoragePort
    Protocol for storage access operations.

Example
-------
```python
from codeintel.core.ports.storage import QueryResult, BatchResult, StoragePort


def handle_result(result: QueryResult) -> int:
    return result.row_count


def write_data(port: StoragePort, rows: list[tuple]) -> BatchResult:
    return port.write_batch("my_table", rows)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class QueryResult:
    """Unified query result for all storage operations.

    This dataclass provides a consistent interface for query results across
    the codebase.

    Attributes
    ----------
    rows
        Sequence of row tuples returned by the query.
    columns
        Column names in result order.
    row_count
        Number of rows returned.

    Examples
    --------
    >>> result = QueryResult.from_rows([("a", 1), ("b", 2)])
    >>> result.row_count
    2
    >>> result.rows
    (('a', 1), ('b', 2))
    """

    rows: tuple[tuple[object, ...], ...] = ()
    columns: tuple[str, ...] = ()
    row_count: int = 0

    @classmethod
    def empty(cls) -> QueryResult:
        """Create an empty query result.

        Returns
        -------
        QueryResult
            Result with no rows.
        """
        return cls(rows=(), columns=(), row_count=0)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[tuple[object, ...]],
        columns: Sequence[str] | None = None,
    ) -> QueryResult:
        """Create a query result from a sequence of rows.

        Parameters
        ----------
        rows
            Row tuples to include in the result.
        columns
            Optional column names for the result.

        Returns
        -------
        QueryResult
            Result containing the provided rows.
        """
        row_tuple = tuple(rows)
        col_tuple = tuple(columns) if columns is not None else ()
        return cls(rows=row_tuple, columns=col_tuple, row_count=len(row_tuple))


@dataclass(frozen=True)
class BatchResult:
    """Unified batch operation result.

    This dataclass provides a consistent interface for batch write operation
    results.

    Attributes
    ----------
    table
        Target table name or key.
    rows_affected
        Number of rows inserted, updated, or deleted.
    success
        Whether the operation succeeded.
    error
        Error message if operation failed.
    duration_s
        Operation duration in seconds.

    Examples
    --------
    >>> result = BatchResult.ok("my_table", 100)
    >>> result.success
    True
    >>> result.rows_affected
    100
    """

    table: str
    rows_affected: int
    success: bool = True
    error: str | None = None
    duration_s: float = 0.0

    @classmethod
    def ok(
        cls,
        table: str,
        rows_affected: int,
        duration_s: float = 0.0,
    ) -> BatchResult:
        """Create a successful batch result.

        Parameters
        ----------
        table
            Target table name.
        rows_affected
            Number of rows affected.
        duration_s
            Operation duration in seconds.

        Returns
        -------
        BatchResult
            Successful result.
        """
        return cls(
            table=table,
            rows_affected=rows_affected,
            success=True,
            duration_s=duration_s,
        )

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
        return cls(table=table, rows_affected=0, success=False, error=error)


@dataclass
class MutableQueryResult:
    """Mutable query result for incremental construction.

    Use this when building results incrementally, then convert to
    immutable QueryResult when complete.

    Attributes
    ----------
    rows
        Mutable list of row tuples.
    columns
        Column names in result order.
    row_count
        Number of rows returned.
    """

    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: tuple[str, ...] = ()
    row_count: int = 0

    def to_query_result(self) -> QueryResult:
        """Convert to immutable QueryResult.

        Returns
        -------
        QueryResult
            Immutable copy of this result.
        """
        return QueryResult(
            rows=tuple(self.rows),
            columns=self.columns,
            row_count=len(self.rows),
        )


@runtime_checkable
class StoragePort(Protocol):
    """Unified storage port protocol for database operations.

    This protocol defines a consistent interface for storage access across
    the codebase. Implementations provide database-agnostic operations for
    queries, batch writes, and deletions.

    Implementations may be backed by DuckDB, SQLite, or other databases
    while maintaining the same interface.
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

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        """Write a batch of rows to a table.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        rows
            Row data matching the table's column order.
        scope
            Optional scope identifier for logging (e.g., "repo@commit").

        Returns
        -------
        BatchResult
            Metadata about the write operation.
        """
        ...

    def delete_by_params(
        self,
        table_key: str,
        params: Sequence[object],
    ) -> int:
        """Delete rows matching the given parameters.

        Uses the table's registered delete statement pattern.

        Parameters
        ----------
        table_key
            Registry table key.
        params
            Parameters for the delete statement.

        Returns
        -------
        int
            Number of rows deleted.
        """
        ...

    def delete_by_paths(
        self,
        table_key: str,
        paths: Sequence[str],
        *,
        path_column: str = "rel_path",
        repo: str | None = None,
        commit: str | None = None,
    ) -> int:
        """Delete rows where path_column matches any of the provided paths.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.docstrings").
        paths
            List of path values to delete.
        path_column
            Name of the column containing paths (default: "rel_path").
        repo
            Optional repository filter when the table includes a repo column.
        commit
            Optional commit filter when the table includes a commit column.

        Returns
        -------
        int
            Number of rows deleted.
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
        """Return the repository root path.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        ...

    def fetch_dataframe(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> object:
        """Execute a query and return results as a DataFrame.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        object
            Query results as a pandas DataFrame.
        """
        ...


__all__ = [
    "BatchResult",
    "MutableQueryResult",
    "QueryResult",
    "StoragePort",
]
