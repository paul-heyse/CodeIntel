"""Fake storage implementations for testing.

This module provides fake implementations of storage protocols for tests
that need deterministic storage behavior without a real database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from tests._helpers.records import CallRecorder, StorageOpCall

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class FakeIngestStorage:
    """In-memory test implementation of IngestStoragePort.

    This fake implements the full ``IngestStoragePort`` protocol with in-memory
    data structures, enabling unit tests to verify ingestion compute logic
    without spinning up a real database.

    Why This Exists
    ---------------
    The ``IngestStoragePort`` protocol exists specifically to enable this kind
    of test isolation. Ingestion compute steps (``AstExtractStep``, ``ScipIngestStep``,
    etc.) depend on the protocol rather than ``DuckDBStorageAdapter`` directly,
    allowing tests to inject this fake for fast, isolated unit testing.

    Attributes
    ----------
    data : dict[str, list[Sequence[object]]]
        In-memory data store keyed by table_key.
    schemas : set[str]
        Set of table keys for which schema has been ensured.
    operations : CallRecorder[StorageOpCall]
        Log of operations for verification (operation_type, table_key, details).

    See Also
    --------
    codeintel.ingestion.ports.storage.IngestStoragePort : Protocol definition
    codeintel.ingestion.adapters.DuckDBStorageAdapter : Production implementation
    """

    data: dict[str, list[Sequence[object]]] = field(default_factory=dict)
    schemas: set[str] = field(default_factory=set)
    operations: CallRecorder[StorageOpCall] = field(default_factory=CallRecorder)

    def ensure_schema(self, table_key: str) -> None:
        """Ensure the schema exists for a table.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        """
        self.schemas.add(table_key)
        if table_key not in self.data:
            self.data[table_key] = []
        self.operations.record(StorageOpCall(op="ensure_schema", target=table_key, details=None))

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
            Optional scope identifier for logging.

        Returns
        -------
        BatchResult
            Metadata about the write operation.
        """
        if table_key not in self.data:
            self.data[table_key] = []
        self.data[table_key].extend(rows)
        self.operations.record(
            StorageOpCall(
                op="write_batch", target=table_key, details={"rows": len(rows), "scope": scope}
            )
        )
        return BatchResult.ok(table_key, len(rows), duration_s=0.0)

    def delete_by_params(
        self,
        table_key: str,
        params: Sequence[object],
    ) -> int:
        """Delete rows matching the given parameters.

        Parameters
        ----------
        table_key
            Registry table key.
        params
            Parameters for the delete statement.

        Returns
        -------
        int
            Number of rows deleted (always 0 in this fake).
        """
        self.operations.record(
            StorageOpCall(op="delete_by_params", target=table_key, details={"params": params})
        )
        return 0

    def delete_by_paths(
        self,
        table_key: str,
        paths: Sequence[str],
        *,
        path_column: str = "rel_path",
    ) -> int:
        """Delete rows where path_column matches any of the provided paths.

        Parameters
        ----------
        table_key
            Registry table key.
        paths
            List of path values to delete.
        path_column
            Name of the column containing paths.

        Returns
        -------
        int
            Number of rows deleted (always 0 in this fake).
        """
        self.operations.record(
            StorageOpCall(
                op="delete_by_paths",
                target=table_key,
                details={"paths": paths, "path_column": path_column},
            )
        )
        return 0

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
            Empty query results (queries not supported in fake).
        """
        self.operations.record(
            StorageOpCall(op="execute_query", target=sql, details={"params": params})
        )
        return QueryResult.empty()

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
            Empty DataFrame-like object.
        """
        self.operations.record(
            StorageOpCall(op="fetch_dataframe", target=sql, details={"params": params})
        )
        return pd.DataFrame()


__all__ = ["FakeIngestStorage"]
