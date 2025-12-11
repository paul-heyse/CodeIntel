"""DuckDB ingestion storage adapter shim (policy backend + ibis).

This compatibility layer keeps the IngestStoragePort surface while routing
all writes/deletes through DuckDBPolicyBackend and reads through the
gateway/ibis connection. Raw SQL helpers are retained only for legacy
query execution in tests; new code should prefer ibis or the policy backend
directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from pandera.errors import SchemaErrors

from codeintel.config.datasets import load_columns_by_table
from codeintel.ingestion.ports.storage import BatchResult, IngestStoragePort, QueryResult
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.ibis_types import and_predicates, ibis_bool, isin_values
from codeintel.storage.pandera_schemas import get_dataset_schema
from codeintel.storage.sql import render_sql
from codeintel.storage.sql.primitives import quote_identifier, quote_table_key

if TYPE_CHECKING:
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
    return render_sql(["DELETE FROM", table_sql, "WHERE", delete_clause])


class DuckDBStorageAdapter(IngestStoragePort):
    """Compatibility shim implementing IngestStoragePort via policy backend."""

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize adapter with a storage gateway."""
        self._gateway = gateway
        self._backend = DuckDBPolicyBackend(gateway)

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
            return BatchResult(table_key=table_key, rows_written=0, duration_s=0.0)
        columns = load_columns_by_table().get(table_key, [])
        inserted = self._backend.bulk_insert(
            table_key,
            [tuple(row) for row in rows],
            columns=columns,
        )
        return BatchResult(table_key=table_key, rows_written=inserted, duration_s=0.0)

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
            table_expr = cast("Any", table)
            table_expr.delete(where=cond)
        except Exception as exc:  # pragma: no cover - delegated to backend
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
        result = self.con.execute(sql, param_list)
        rows = result.fetchall()
        columns = tuple(desc[0] for desc in result.description) if result.description else ()
        return QueryResult(rows=list(rows), columns=columns, row_count=len(rows))

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
        return self.con.execute(sql, param_list).fetch_df()


@dataclass
class IngestStorageService:
    """Validate and write ingest batches using a storage port."""

    storage: IngestStoragePort
    validate: bool = True

    def run_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        delete_params: Sequence[object] | None = None,
        scope: str | None = None,
    ) -> BatchResult:
        """Write batch with optional pre-delete and Pandera validation.

        Returns
        -------
        BatchResult
            Result containing rows written and duration.
        """
        self.storage.ensure_schema(table_key)

        validated_rows = rows
        if self.validate and rows:
            validated_rows = self._validate_rows(table_key, rows)

        if delete_params is not None:
            self.storage.delete_by_params(table_key, delete_params)

        return self.storage.write_batch(table_key, validated_rows, scope=scope)

    @staticmethod
    def _validate_rows(
        table_key: str, rows: Sequence[Sequence[object]]
    ) -> Sequence[Sequence[object]]:
        """Validate rows using Pandera schema if available.

        Returns
        -------
        Sequence[Sequence[object]]
            Original rows irrespective of validation outcome.
        """
        schema = get_dataset_schema(table_key)
        if schema is None:
            return rows

        registry_cols = load_columns_by_table().get(table_key)
        if registry_cols is None:
            return rows

        df = pd.DataFrame([list(row) for row in rows], columns=pd.Index(registry_cols))
        try:
            schema.validate(df, lazy=True)
        except SchemaErrors as exc:  # pragma: no cover - advisory path
            log.warning(
                "Pandera validation warning for %s: %s",
                table_key,
                str(exc)[:200],
            )
        return rows

    @classmethod
    def from_gateway(
        cls, gateway: StorageGateway, *, validate: bool = True
    ) -> IngestStorageService:
        """Create a service instance from a StorageGateway.

        Returns
        -------
        IngestStorageService
            New service instance wrapping the provided gateway.
        """
        return cls(storage=DuckDBStorageAdapter(gateway), validate=validate)


__all__ = [
    "SNAPSHOT_PARAM_LEN",
    "DuckDBStorageAdapter",
    "IngestStorageService",
    "build_delete_in_query",
    "quote_identifier",
    "quote_table_key",
]
