"""DuckDB storage adapter implementing IngestStoragePort.

This module provides the production implementation of ``IngestStoragePort``
for the ingestion layer. The adapter routes operations through the established
storage infrastructure:

- Writes/deletes: ``DuckDBPolicyBackend`` for type-safe bulk operations
- Reads: ``StorageGateway`` for query execution and DataFrame retrieval
- Schema management: ``DuckDBPolicyBackend.ensure_schemas_preserve()``

Why This Adapter Exists
-----------------------
The ``IngestStoragePort`` protocol enables test isolation - ingestion compute
steps can be unit tested using ``FakeIngestStorage`` without a database. This
adapter is the production implementation that connects to real storage.

See Also
--------
codeintel.ingestion.ports.storage.IngestStoragePort : Protocol definition
tests._helpers.fakes.storage.FakeIngestStorage : Test implementation
codeintel.storage.duckdb_policy_backend.DuckDBPolicyBackend : Underlying backend
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.config.datasets import load_columns_by_table
from codeintel.ingestion.ports.storage import BatchResult, IngestStoragePort, QueryResult
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.ibis_types import and_predicates, ibis_bool, isin_values

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)
SNAPSHOT_PARAM_LEN = 2


def build_delete_in_query(table_sql: str, column_sql: str, count: int) -> str:
    """Return a parameterized DELETE ... IN statement.

    Returns
    -------
    str
        Rendered SQL string containing placeholders.
    """
    placeholders = ", ".join(["?"] * count)
    delete_clause = f"{column_sql} IN ({placeholders})"
    return " ".join(part for part in ("DELETE FROM", table_sql, "WHERE", delete_clause) if part)


class DuckDBStorageAdapter(IngestStoragePort):
    """Production storage adapter implementing IngestStoragePort.

    This adapter wraps ``StorageGateway`` and ``DuckDBPolicyBackend`` to provide
    the ``IngestStoragePort`` interface for ingestion compute steps.

    The abstraction exists primarily for **testability**: ingestion compute steps
    depend on ``IngestStoragePort`` rather than concrete storage, allowing tests
    to inject ``FakeIngestStorage`` for unit testing without a database.

    Parameters
    ----------
    gateway
        Storage gateway providing database connection and Ibis interface.

    Examples
    --------
    >>> storage = DuckDBStorageAdapter(ctx.gateway)
    >>> result = storage.write_batch("core.modules", rows)
    >>> print(f"Wrote {result.rows_written} rows")
    """

    ADAPTER_NAME: ClassVar[str] = "duckdb_storage"

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize adapter with a storage gateway."""
        self._gateway = gateway
        self._backend = DuckDBPolicyBackend(gateway)

    def initialize(self) -> None:
        """Initialize the adapter (no-op, gateway is passed in constructor)."""

    def close(self) -> None:
        """Close the adapter (no-op, does not own gateway lifecycle)."""

    @property
    def is_available(self) -> bool:
        """Check if adapter is available.

        Returns
        -------
        bool
            True if gateway is available.
        """
        return self._gateway is not None

    @property
    def con(self) -> DuckDBConnection:
        """Return underlying DuckDB connection."""
        return self._gateway.con

    @staticmethod
    def _validate_table_exists(table_key: str) -> None:
        """Raise when table_key is not present in the dataset registry.

        Raises
        ------
        RuntimeError
            If the table key is not registered.
        """
        if table_key not in load_columns_by_table():
            message = f"Table {table_key} missing from TABLE_SCHEMAS"
            raise RuntimeError(message)

    def ensure_schema(self, table_key: str) -> None:
        """Ensure schemas exist (idempotent)."""
        self._validate_table_exists(table_key)
        self._backend.ensure_schemas_preserve()

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        """Insert rows using policy backend bulk_insert.

        Returns
        -------
        BatchResult
            Result including rows written.
        """
        _ = scope
        self._validate_table_exists(table_key)
        if not rows:
            return BatchResult.from_write(table_key=table_key, rows_written=0, duration_s=0.0)
        columns = load_columns_by_table().get(table_key, [])
        inserted = self._backend.bulk_insert(
            table_key,
            [tuple(row) for row in rows],
            columns=columns,
        )
        return BatchResult.from_write(table_key=table_key, rows_written=inserted, duration_s=0.0)

    def delete_by_params(
        self,
        table_key: str,
        params: Sequence[object],
    ) -> int:
        """Delete snapshot rows when repo/commit are provided.

        Returns
        -------
        int
            Number of deleted rows (DuckDB returns 0).
        """
        self._validate_table_exists(table_key)
        if len(params) == SNAPSHOT_PARAM_LEN:
            self._backend.delete_for_snapshot(table_key, repo=str(params[0]), commit=str(params[1]))
        return 0

    def delete_by_paths(
        self,
        table_key: str,
        paths: Sequence[str],
        *,
        path_column: str = "rel_path",
        repo: str | None = None,
        commit: str | None = None,
    ) -> int:
        """Delete rows filtered by path and optional repo/commit via ibis.

        Returns
        -------
        int
            Number of deleted rows (DuckDB returns 0).

        Raises
        ------
        ValueError
            If deletion fails.
        """
        self._validate_table_exists(table_key)
        if not paths:
            return 0

        table = self._gateway.ibis.table(table_key)
        predicates = [isin_values(table[path_column], list(paths))]
        schema_names_attr = table.schema().names
        schema_names = list(cast("Sequence[str]", schema_names_attr))
        if repo is not None and "repo" in schema_names:
            predicates.append(ibis_bool(table["repo"] == repo))
        if commit is not None and "commit" in schema_names:
            predicates.append(ibis_bool(table["commit"] == commit))
        cond = and_predicates(*predicates)

        try:
            self._gateway.ibis.delete(table_key, where=cond)
        except Exception as exc:
            message = f"Failed to delete rows from {table_key}"
            raise ValueError(message) from exc
        return 0

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a raw SQL query and return a structured result.

        Returns
        -------
        QueryResult
            Container holding rows, columns, and row count.
        """
        param_list = list(params) if params else []
        result = self._gateway.execute(sql, param_list)
        rows = result.fetchall()
        columns = tuple(desc[0] for desc in result.description) if result.description else ()
        return QueryResult.from_rows([tuple(row) for row in rows], columns=columns)

    def fetch_dataframe(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> pd.DataFrame:
        """Execute a query and return results as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Resulting dataframe from the query execution.
        """
        param_list = list(params) if params else []
        return self._gateway.execute(sql, param_list).fetch_df()


__all__ = ["SNAPSHOT_PARAM_LEN", "DuckDBStorageAdapter", "build_delete_in_query"]
