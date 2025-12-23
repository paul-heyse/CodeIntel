"""Storage port protocol for ingestion data persistence.

This module re-exports unified storage types from ``codeintel.core.ports.storage``
to provide domain-appropriate imports for ingestion code.

The protocol abstracts database-specific operations like schema management,
batch writes, and queries.

See Also
--------
codeintel.core.ports.storage : Canonical storage types
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.core.ports.storage import BatchResult, MutableQueryResult, QueryResult, StoragePort

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class IngestStoragePort(Protocol):
    """Port protocol for persisting ingestion data.

    This protocol defines the contract for all storage operations in the
    ingestion layer, enabling:

    1. **Test isolation**: ``FakeIngestStorage`` implements this protocol
       for unit testing ingestion compute steps without a database.
    2. **Consistent return types**: All write operations return ``BatchResult``
       with uniform access to ``rows_affected`` and ``table``.
    3. **Clear architectural boundary**: Separates ingestion compute logic
       from database implementation details.

    Implementations
    ---------------
    - ``DuckDBStorageAdapter``: Production adapter using ``StorageGateway``
      and ``DuckDBPolicyBackend`` for actual database operations.
    - ``FakeIngestStorage``: Test fake with in-memory storage and operation
      recording for verification.

    Why This Abstraction Exists
    ---------------------------
    The ingestion compute steps (``AstExtractStep``, ``DocstringsExtractStep``,
    etc.) need to write data to storage. By depending on this protocol rather
    than concrete implementations, the compute logic can be tested in isolation
    using ``FakeIngestStorage`` without spinning up a database.

    See Also
    --------
    codeintel.ingestion.adapters.DuckDBStorageAdapter : Production implementation
    tests._helpers.fakes.storage.FakeIngestStorage : Test implementation
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
    "MutableQueryResult",
    "QueryResult",
    "StoragePort",
]
