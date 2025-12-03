"""DuckDB storage adapter implementing IngestStoragePort.

This adapter provides DuckDB-specific storage operations including
macro-based batch inserts, schema management, and query execution.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from duckdb import Error as DuckDBError

from codeintel.config.datasets import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    load_columns_by_table,
)
from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from codeintel.storage.ingest_macros import (
    assert_ingest_macros_present,
    ensure_ingest_macros,
    list_ingest_macros,
)
from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.sql_builder import render_sql
from codeintel.storage.sql_helpers import (
    ensure_schema as _ensure_schema,
)
from codeintel.storage.sql_helpers import (
    prepared_statements_dynamic,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)

# Batch size threshold below which macro overhead exceeds benefits
SMALL_BATCH_THRESHOLD = 25

# Regex for validating SQL identifiers
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Mapping of table keys to ingest macro names
INGEST_MACROS: dict[str, str] = {
    table_key: f"metadata.ingest_{table_key.split('.', maxsplit=1)[1]}"
    for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
    if not table_key.startswith("metadata.") and contract.schema is not None
}

# Cache for macro names per connection
_MACRO_CACHE: dict[int, set[str]] = {}


def _quote_identifier(identifier: str) -> str:
    """Return a safely quoted SQL identifier.

    Returns
    -------
    str
        Quoted identifier.

    Raises
    ------
    ValueError
        If the identifier contains unsafe characters.
    """
    if not _IDENTIFIER_RE.match(identifier):
        message = f"Unsafe identifier: {identifier!r}"
        raise ValueError(message)
    return f'"{identifier}"'


def _quote_table_key(table_key: str) -> tuple[str, str, str]:
    """Quote schema and table components after validating against registry.

    Returns
    -------
    tuple[str, str, str]
        Schema name, table name, and fully quoted identifier.

    Raises
    ------
    ValueError
        If table key is unknown or components are unsafe.
    """
    if table_key not in DATASET_CONTRACTS_BY_TABLE_KEY:
        message = f"Unknown table key: {table_key}"
        raise ValueError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    quoted = f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"
    return schema_name, table_name, quoted


def _build_delete_in_query(table_sql: str, column_sql: str, count: int) -> str:
    """Build a DELETE IN query with validated identifiers.

    This function assumes table_sql and column_sql have already been validated
    and quoted by _quote_table_key and _quote_identifier respectively.

    Parameters
    ----------
    table_sql
        Quoted table name (e.g., '"schema"."table"').
    column_sql
        Quoted column name (e.g., '"path"').
    count
        Number of placeholder values.

    Returns
    -------
    str
        DELETE query string with ? placeholders.
    """
    placeholders = ", ".join(["?"] * count)
    delete_clause = f"{column_sql} IN ({placeholders})"
    return render_sql(["DELETE FROM", table_sql, "WHERE", delete_clause])


def _quote_macro_name(macro_name: str) -> str:
    """Return a validated macro identifier (optionally schema-qualified).

    Returns
    -------
    str
        Validated macro name.

    Raises
    ------
    ValueError
        If any macro name component is unsafe.
    """
    parts = macro_name.split(".")
    for part in parts:
        if not part:
            message = f"Unsafe macro name: {macro_name!r}"
            raise ValueError(message)
        _quote_identifier(part)
    return ".".join(part for part in parts if part)


def quote_identifier(identifier: str) -> str:
    """Public wrapper for quoting SQL identifiers.

    Returns
    -------
    str
        Quoted identifier suitable for SQL statements.
    """
    return _quote_identifier(identifier)


def quote_table_key(table_key: str) -> tuple[str, str, str]:
    """Public wrapper for quoting validated table keys.

    Returns
    -------
    tuple[str, str, str]
        Schema name, table name, and fully quoted identifier.
    """
    return _quote_table_key(table_key)


def build_delete_in_query(table_sql: str, column_sql: str, count: int) -> str:
    """Public wrapper for building delete queries with IN clauses.

    Returns
    -------
    str
        Parameterized DELETE statement.
    """
    return _build_delete_in_query(table_sql, column_sql, count)


def quote_macro_name(macro_name: str) -> str:
    """Public wrapper for quoting macro names safely.

    Returns
    -------
    str
        Validated macro name.
    """
    return _quote_macro_name(macro_name)


def _load_macro_names(con: DuckDBConnection) -> set[str]:
    """Return macro names (qualified + unqualified) for the active connection.

    Returns
    -------
    set[str]
        Set of macro names available on the connection.
    """
    rows = con.execute(
        """
        SELECT schema_name, function_name
        FROM duckdb_functions()
        WHERE function_type IN ('macro', 'table_macro')
        """
    ).fetchall()
    names: set[str] = set()
    for schema_name, function_name in rows:
        fn = str(function_name)
        names.add(fn.lower())
        if schema_name is not None:
            names.add(f"{schema_name}.{fn}".lower())
    return names


def _assert_macro_available(con: DuckDBConnection, macro_name: str) -> bool:
    """Ensure a specific ingest macro is available on the connection.

    Returns
    -------
    bool
        True if macro is available, False otherwise.
    """
    ensure_ingest_macros(con)
    _MACRO_CACHE.pop(id(con), None)
    target = macro_name.lower()
    macros = list_ingest_macros(con)
    short = target.split(".")[-1]
    if target in macros or short in macros:
        return True
    # Retry once after forcing registration again.
    ensure_ingest_macros(con)
    _MACRO_CACHE.pop(id(con), None)
    macros = list_ingest_macros(con)
    return target in macros or short in macros


class DuckDBStorageAdapter:
    """DuckDB storage adapter implementing IngestStoragePort.

    This adapter provides DuckDB-specific storage operations including
    macro-based batch inserts for performance, schema management, and
    query execution.

    Parameters
    ----------
    gateway
        StorageGateway providing DuckDB connection.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize the adapter with a storage gateway.

        Parameters
        ----------
        gateway
            StorageGateway providing DuckDB connection.
        """
        self._gateway = gateway

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Active database connection.
        """
        return self._gateway.con

    def ensure_schema(self, table_key: str) -> None:
        """Ensure the schema exists for a table.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        """
        _ensure_schema(self.con, table_key)

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        """Write a batch of rows to a table.

        Uses macro-based insertion for large batches and falls back to
        prepared statements for small batches.

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
        start = time.perf_counter()

        if not rows:
            return BatchResult(table_key=table_key, rows_written=0, duration_s=0.0)

        rows_inserted = self._ingest_via_macro(table_key, rows)
        duration = time.perf_counter() - start

        if scope is not None:
            log.info(
                "ingest scope=%s table=%s rows=%d duration=%.2fs",
                scope,
                table_key,
                rows_inserted,
                duration,
            )
        else:
            log.info("ingest table=%s rows=%d duration=%.2fs", table_key, rows_inserted, duration)

        return BatchResult(table_key=table_key, rows_written=rows_inserted, duration_s=duration)

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
            Number of rows deleted.
        """
        self.ensure_schema(table_key)
        stmts = prepared_statements_dynamic(self.con, table_key)

        if stmts.delete_sql is None:
            return 0

        self.con.execute(stmts.delete_sql, list(params))
        return 0  # DuckDB doesn't return affected row count easily

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
            Registry table key (e.g., "core.docstrings").
        paths
            List of path values to delete.
        path_column
            Name of the column containing paths (default: "rel_path").

        Returns
        -------
        int
            Number of rows deleted.
        """
        if not paths:
            return 0

        self.ensure_schema(table_key)
        # Use validated identifiers to construct safe SQL
        _, _, table_sql = _quote_table_key(table_key)
        safe_column = _quote_identifier(path_column)
        delete_sql = _build_delete_in_query(table_sql, safe_column, len(paths))
        self.con.execute(delete_sql, list(paths))
        return 0  # DuckDB doesn't return affected row count easily

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

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame.
        """
        param_list = list(params) if params else []
        return self.con.execute(sql, param_list).fetch_df()

    def _ingest_via_macro(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
    ) -> int:
        """Insert rows using macro-backed path when available.

        Returns
        -------
        int
            Number of rows inserted.

        Raises
        ------
        ValueError
            If table identifiers are unsafe.
        """
        registry_cols, macro_name = self._prepare_registry(table_key)

        # For small batches, use prepared insert directly
        if len(rows) <= SMALL_BATCH_THRESHOLD:
            return self._fallback_prepared_insert(table_key, registry_cols, rows)

        if not _assert_macro_available(self.con, macro_name):
            apply_all_schemas(self.con)
            log.warning(
                "Falling back to prepared insert; macro missing",
                extra={"table_key": table_key, "macro_name": macro_name},
            )
            try:
                return self._fallback_prepared_insert(table_key, registry_cols, rows)
            except DuckDBError:
                apply_all_schemas(self.con)
                return self._fallback_prepared_insert(table_key, registry_cols, rows)

        try:
            _, _, table_sql = _quote_table_key(table_key)
            safe_macro = _quote_macro_name(macro_name)
        except ValueError as exc:
            message = f"Unsafe identifiers for ingest table {table_key}"
            raise ValueError(message) from exc

        schema_rel = self.con.table(table_sql)
        self.con.execute("DROP TABLE IF EXISTS temp_ingest_values")
        schema_rel.limit(0).create("temp_ingest_values")
        df = pd.DataFrame([tuple(row) for row in rows], columns=pd.Index(registry_cols))
        self.con.append("temp_ingest_values", df, by_name=True)

        try:
            macro_rel = self.con.sql(
                "".join(("SELECT * FROM ", safe_macro, "('temp_ingest_values')")),
            )
            macro_rel.insert_into(table_sql)
            return len(rows)
        except DuckDBError:
            log.warning(
                "Macro unavailable at execution time; falling back to prepared insert",
                extra={"table_key": table_key, "macro_name": macro_name},
                exc_info=True,
            )
            return self._fallback_prepared_insert(table_key, registry_cols, rows)

    def _prepare_registry(self, table_key: str) -> tuple[list[str], str]:
        """Ensure schemas/macros exist and return registry columns plus macro name.

        Returns
        -------
        tuple[list[str], str]
            Registry columns and macro name.

        Raises
        ------
        RuntimeError
            If registry metadata is missing.
        """
        ensure_ingest_macros(self.con)
        assert_ingest_macros_present(self.con)
        try:
            self.ensure_schema(table_key)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "missing" not in message:
                raise
            apply_all_schemas(self.con)
            self.ensure_schema(table_key)
        registry_cols = load_columns_by_table().get(table_key)
        if registry_cols is None:
            message = f"Table {table_key} missing from registry"
            raise RuntimeError(message)
        macro_name = INGEST_MACROS.get(table_key)
        if macro_name is None:
            message = f"No ingest macro is defined for table {table_key}"
            raise RuntimeError(message)
        return registry_cols, macro_name

    def _fallback_prepared_insert(
        self,
        table_key: str,
        registry_cols: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> int:
        """Fallback path when macros are unavailable.

        Returns
        -------
        int
            Number of rows inserted.
        """
        _, _, table_sql = _quote_table_key(table_key)
        df = pd.DataFrame([tuple(row) for row in rows], columns=pd.Index(registry_cols))
        try:
            self.con.from_df(df).insert_into(table_sql)
        except DuckDBError:
            apply_all_schemas(self.con)
            self.con.from_df(df).insert_into(table_sql)
        return len(rows)


__all__ = [
    "DuckDBStorageAdapter",
    "build_delete_in_query",
    "quote_identifier",
    "quote_macro_name",
    "quote_table_key",
]
