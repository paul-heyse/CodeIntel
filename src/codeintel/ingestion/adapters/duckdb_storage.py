"""DuckDB storage adapter implementing IngestStoragePort.

This module provides the production implementation of ``IngestStoragePort``
for the ingestion layer. The adapter routes operations through the established
storage infrastructure:

- Writes/deletes: ``DuckDBPolicyBackend`` for type-safe bulk operations
- Reads: ``StorageGateway`` for query execution and Arrow batch retrieval
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

from codeintel.config.datasets.columns import load_columns_by_table
from codeintel.core.schemas.service import get_schema_service
from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import iter_tuples_from_arrow_reader

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)
SNAPSHOT_PARAM_LEN = 2


def _schema_columns_for_table(table_key: str) -> list[str] | None:
    try:
        service = get_schema_service()
    except RuntimeError:
        return None
    schema = service.get_table_schema(table_key)
    if schema is None:
        return None
    return list(schema.column_names())


def _columns_for_table(table_key: str) -> list[str] | None:
    schema_columns = _schema_columns_for_table(table_key)
    if schema_columns is not None:
        return schema_columns
    return load_columns_by_table().get(table_key)


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


class DuckDBStorageAdapter:
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
    >>> print(f"Wrote {result.rows_affected} rows")
    """

    ADAPTER_NAME: ClassVar[str] = "duckdb_storage"

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize adapter with a storage gateway."""
        self._gateway = gateway
        self._backend = gateway.policy

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
        columns = _columns_for_table(table_key)
        if columns is None:
            message = f"Table {table_key} missing from schema registry"
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

        Raises
        ------
        RuntimeError
            If the table key is missing from the schema registry.
        """
        _ = scope
        self._validate_table_exists(table_key)
        if not rows:
            return BatchResult.ok(table_key, 0, duration_s=0.0)
        columns = _columns_for_table(table_key)
        if columns is None:
            message = f"Table {table_key} missing from schema registry"
            raise RuntimeError(message)
        inserted = self._backend.bulk_insert(
            table_key,
            [tuple(row) for row in rows],
            columns=columns,
        )
        return BatchResult.ok(table_key, inserted, duration_s=0.0)

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
        """Delete rows filtered by path and optional repo/commit via DuckDB relations.

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
        if not path_column.isidentifier():
            message = f"Invalid path column: {path_column}"
            raise ValueError(message)

        relation = self._gateway.relation_from_table_key(table_key)
        schema_names = list(cast("Sequence[str]", relation.columns))
        conditions: list[str] = []
        params: list[object] = []

        placeholders = ", ".join(["?"] * len(paths))
        conditions.append(f"{path_column} IN ({placeholders})")
        params.extend(paths)
        if repo is not None and "repo" in schema_names:
            conditions.append("repo = ?")
            params.append(repo)
        if commit is not None and "commit" in schema_names:
            conditions.append("commit = ?")
            params.append(commit)

        where_clause = " AND ".join(conditions)
        sql = f"DELETE FROM {table_key} WHERE {where_clause}"
        try:
            self._gateway.execute(sql, params)
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
        reader = result.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        rows = list(iter_tuples_from_arrow_reader(reader))
        columns = tuple(desc[0] for desc in result.description) if result.description else ()
        return QueryResult.from_rows([tuple(row) for row in rows], columns=columns)

    def fetch_arrow_reader(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        batch_size: int | None = None,
    ) -> pa.RecordBatchReader:
        """Execute a query and return results as a record batch reader.

        Returns
        -------
        pa.RecordBatchReader
            Arrow record batch reader for the query results.
        """
        param_list = list(params) if params else []
        resolved_batch_size = batch_size or DEFAULT_ARROW_BATCH_SIZE
        return self._gateway.execute(sql, param_list).fetch_record_batch(resolved_batch_size)


__all__ = ["SNAPSHOT_PARAM_LEN", "DuckDBStorageAdapter", "build_delete_in_query"]
