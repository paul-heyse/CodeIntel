"""Connection-aware schema helpers driven by the DuckDB registry metadata."""

from __future__ import annotations

from dataclasses import dataclass

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns

_INGESTION_COLUMNS_VERIFIED: list[bool] = [False]


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete SQL for a table (registry-driven)."""

    insert_sql: str
    delete_sql: str | None = None


def prepared_statements_dynamic(
    con: DuckDBPyConnection,
    table_key: str,
) -> PreparedStatements:
    """
    Return prepared SQL using registry-derived column order for a table.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Registry key (e.g., "core.ast_nodes", "analytics.function_metrics").

    Returns
    -------
    PreparedStatements
        Insert (and optional delete) SQL with column order sourced from the
        DuckDB registry via `build_registry_contracts`.

    Raises
    ------
    RuntimeError
        If the table is missing from the registry.
    """
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)

    cols_sql = ", ".join(registry_cols)
    placeholders = ", ".join("?" for _ in registry_cols)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'

    insert_sql = (
        f"INSERT INTO {table_sql} ({cols_sql}) VALUES ({placeholders})"  # noqa: S608
    )
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
    )


def ensure_schema(con: DuckDBPyConnection, table_key: str) -> None:
    """
    Validate that the live DuckDB table matches the registry definition.

    This:
    - Ensures that the literal column lists in `ingestion_sql` haven't drifted
      from the registry (once per process).
    - Ensures that the DuckDB table's columns & order match the registry.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Fully qualified table name (schema.table).

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    if not _INGESTION_COLUMNS_VERIFIED[0]:
        verify_ingestion_columns(con)
        _INGESTION_COLUMNS_VERIFIED[0] = True

    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)

    schema_name, table_name = table_key.split(".", maxsplit=1)
    info = con.execute(f"PRAGMA table_info({schema_name}.{table_name})").fetchall()
    if not info:
        message = f"Table {table_key} is missing"
        raise RuntimeError(message)

    names = [row[1] for row in info]
    expected_cols = registry_cols
    if names != expected_cols:
        message = f"Column order mismatch for {table_key}: db={names}, registry={expected_cols}"
        raise RuntimeError(message)


__all__ = ["PreparedStatements", "ensure_schema", "prepared_statements_dynamic"]
