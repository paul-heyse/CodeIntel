"""Storage port protocol for ingestion data persistence.

This module re-exports unified storage types from ``codeintel.core.ports.storage``
with backward-compatible aliases for the ingestion naming convention.

The protocol abstracts database-specific operations like schema management,
batch writes, and queries.

See Also
--------
codeintel.core.ports.storage : Canonical storage types
codeintel.core.ports.BaseQueryResult : Base protocol for query results
codeintel.core.ports.BaseBatchResult : Base protocol for batch results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codeintel.core.ports.storage import BatchResult as CoreBatchResult

# Re-export core types
from codeintel.core.ports.storage import (
    MutableQueryResult,
    StoragePort,
)
from codeintel.core.ports.storage import QueryResult as CoreQueryResult

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class BatchResult:
    """Result metadata for a batch write operation.

    This class provides backward compatibility with the ingestion naming
    convention (table_key, rows_written) while delegating to the core
    BatchResult implementation.

    Attributes
    ----------
    table_key
        Registry table key (e.g., "core.ast_nodes").
    rows_written
        Number of rows successfully written.
    duration_s
        Operation duration in seconds.

    Notes
    -----
    For forward compatibility with core, this class provides aliases:
    - ``table`` is an alias for ``table_key``
    - ``rows_affected`` is an alias for ``rows_written``
    """

    table_key: str
    rows_written: int
    duration_s: float = 0.0

    @property
    def table(self) -> str:
        """Alias for table_key (core compatibility).

        Returns
        -------
        str
            Target table name or key.
        """
        return self.table_key

    @property
    def rows_affected(self) -> int:
        """Alias for rows_written (core compatibility).

        Returns
        -------
        int
            Number of rows affected.
        """
        return self.rows_written

    @property
    def success(self) -> bool:
        """Operation success indicator.

        Returns
        -------
        bool
            Always True for ingestion results (errors raise exceptions).
        """
        return True

    @property
    def error(self) -> str | None:
        """Error message (always None for successful results).

        Returns
        -------
        str | None
            Always None.
        """
        return None

    def to_core(self) -> CoreBatchResult:
        """Convert to core BatchResult.

        Returns
        -------
        CoreBatchResult
            Core-compatible batch result.
        """
        return CoreBatchResult(
            table=self.table_key,
            rows_affected=self.rows_written,
            success=True,
            duration_s=self.duration_s,
        )


@dataclass
class QueryResult:
    """Result from a query operation.

    This class provides backward compatibility with the ingestion interface
    while maintaining compatibility with the core QueryResult.

    Attributes
    ----------
    rows
        Query result rows (mutable list for incremental construction).
    columns
        Column names in result order.
    row_count
        Number of rows returned.
    """

    rows: list[tuple[Any, ...]] = field(default_factory=list)
    columns: tuple[str, ...] = ()
    row_count: int = 0

    def to_core(self) -> CoreQueryResult:
        """Convert to core QueryResult.

        Returns
        -------
        CoreQueryResult
            Core-compatible query result.
        """
        return CoreQueryResult(
            rows=tuple(self.rows),
            columns=self.columns,
            row_count=len(self.rows),
        )

    @classmethod
    def from_core(cls, core_result: CoreQueryResult) -> QueryResult:
        """Create from a core QueryResult.

        Parameters
        ----------
        core_result
            Core query result to convert.

        Returns
        -------
        QueryResult
            Ingestion-compatible query result.
        """
        return cls(
            rows=list(core_result.rows),
            columns=core_result.columns,
            row_count=core_result.row_count,
        )


@runtime_checkable
class IngestStoragePort(Protocol):
    """Port protocol for persisting ingestion data.

    This protocol defines the contract for all storage operations in the
    ingestion layer, enabling:

    1. **Test isolation**: ``FakeIngestStorage`` implements this protocol
       for unit testing ingestion compute steps without a database.
    2. **Consistent return types**: All write operations return ``BatchResult``
       with uniform access to ``rows_written``, ``table_key``, etc.
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
    The ingestion compute steps (``AstExtractStep``, ``ScipIngestStep``, etc.)
    need to write data to storage. By depending on this protocol rather than
    concrete implementations, the compute logic can be tested in isolation
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
    "CoreBatchResult",
    "CoreQueryResult",
    "IngestStoragePort",
    "MutableQueryResult",
    "QueryResult",
    "StoragePort",
]
