"""DuckDB storage adapter shim using policy backend and ibis.

This adapter implements IngestStoragePort by delegating writes/deletes
to DuckDBPolicyBackend and reads to the gateway/ibis. It exists as a
compatibility layer while ingestion steps are refactored toward direct
policy-backend usage.
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
from codeintel.storage.metadata import INGEST_MACROS
from codeintel.storage.pandera_schemas import get_dataset_schema
from codeintel.storage.sql import render_sql
from codeintel.storage.sql.primitives import (
    quote_identifier,
    quote_table_key,
)
from codeintel.storage.ibis_types import and_predicates, ibis_bool

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)
SNAPSHOT_PARAM_LEN = 2
SMALL_BATCH_THRESHOLD = 25

__all__ = [
    "DuckDBStorageAdapter",
    "INGEST_MACROS",
    "SMALL_BATCH_THRESHOLD",
    "build_delete_in_query",
    "quote_identifier",
    "quote_macro_name",
    "quote_table_key",
]


def build_delete_in_query(table_sql: str, column_sql: str, count: int) -> str:
    """Build parameterized DELETE statement with an IN clause."""
    placeholders = ", ".join(["?"] * count)
    delete_clause = f"{column_sql} IN ({placeholders})"
    return render_sql(["DELETE FROM", table_sql, "WHERE", delete_clause])


def _quote_macro_name(macro_name: str) -> str:
    """Return a validated macro identifier (optionally schema-qualified)."""
    parts = macro_name.split(".")
    if not parts or any(not part for part in parts):
        message = f"Unsafe macro name: {macro_name}"
        raise ValueError(message)
    return ".".join(parts)


def quote_macro_name(macro_name: str) -> str:
    """Public wrapper for quoting macro names safely."""
    return _quote_macro_name(macro_name)


class DuckDBStorageAdapter(IngestStoragePort):
    """Adapter implementing ingestion storage using DuckDB."""

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize adapter with a storage gateway."""
        self._gateway = gateway
        self._backend = DuckDBPolicyBackend(gateway)

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self._gateway.con

    def ensure_schema(self, table_key: str) -> None:
        """Ensure schemas are applied for the given table."""
        _ = table_key
        self._backend.ensure_schemas_preserve()

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        """Insert a batch of rows into the given table."""
        _ = scope
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
        """Delete rows by snapshot parameters."""
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
        """Delete rows by path with optional repo/commit filters."""
        if not paths:
            return 0

        table = cast("Any", self._gateway.ibis.table(table_key))
        predicates = [ibis_bool(cast("Any", table[path_column]).isin(list(paths)))]
        if repo is not None and "repo" in table.schema().names:
            predicates.append(ibis_bool(table["repo"] == repo))
        if commit is not None and "commit" in table.schema().names:
            predicates.append(ibis_bool(table["commit"] == commit))
        cond = and_predicates(*predicates)

        try:
            table.delete(where=cond)
        except Exception as exc:  # pragma: no cover - delegated to backend
            message = f"Failed to delete rows from {table_key}"
            raise ValueError(message) from exc
        return 0

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a raw SQL query and return a structured result."""
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
        param_list = list(params) if params else []
        return self.con.execute(sql, param_list).fetch_df()


@dataclass
class IngestStorageService:
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
        return cls(storage=DuckDBStorageAdapter(gateway), validate=validate)


__all__ = [
    "DuckDBStorageAdapter",
    "IngestStorageService",
    "build_delete_in_query",
    "quote_identifier",
    "quote_table_key",
]
