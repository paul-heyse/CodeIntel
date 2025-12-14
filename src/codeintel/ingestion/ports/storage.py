"""Storage port protocol for ingestion data persistence.

This module defines the port protocol for persisting ingestion data to storage.
The protocol abstracts database-specific operations like schema management,
batch writes, and queries.

See Also
--------
codeintel.core.ports.BaseQueryResult : Base protocol for query results
codeintel.core.ports.BaseBatchResult : Base protocol for batch results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

# Note: BatchResult and QueryResult implement the protocols from
# codeintel.core.ports (BaseBatchResult and BaseQueryResult respectively)


@dataclass(frozen=True)
class BatchResult:
    """Result metadata for a batch write operation.

    Attributes
    ----------
    table_key
        Registry table key (e.g., "core.ast_nodes").
    rows_written
        Number of rows successfully written.
    duration_s
        Operation duration in seconds.
    """

    table_key: str
    rows_written: int
    duration_s: float = 0.0


@dataclass
class QueryResult:
    """Result from a query operation.

    Attributes
    ----------
    rows
        Query result rows.
    columns
        Column names in result order.
    row_count
        Number of rows returned.
    """

    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: tuple[str, ...] = ()
    row_count: int = 0


@runtime_checkable
class IngestStoragePort(Protocol):
    """Port protocol for persisting ingestion data.

    This protocol abstracts storage operations to enable testing with
    mock implementations and potential future database swaps.

    Implementations should handle:
    - Schema creation and validation
    - Batch row insertion with appropriate batching strategy
    - Row deletion for incremental updates
    - Query execution for data retrieval
    """

    def ensure_schema(self, table_key: str) -> None:
        """Ensure the schema exists for a table.

        Create the table schema if it does not exist, or validate that the
        existing schema matches the expected definition.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").

        Raises
        ------
        RuntimeError
            If schema validation fails.
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

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a query and return results.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        QueryResult
            Query results with rows and metadata.
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
    "IngestStoragePort",
    "QueryResult",
]
