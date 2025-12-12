"""Schema-aware SQL query builder functions.

This module provides functions that integrate with codeintel.config.datasets
for schema-aware SQL operations.

For basic SQL building without schema dependencies, use sql.primitives.

Note: Import this module directly (from codeintel.storage.sql.builder import ...)
rather than from the sql package to avoid circular imports with config.datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    TABLE_SCHEMAS,
)
from codeintel.storage.sql.primitives import (
    PreparedStatements,
    build_insert_sql,
    quote_table_key,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = [
    "ensure_schema",
    "prepared_statements_dynamic",
]


def prepared_statements_dynamic(
    _unused_con: DuckDBPyConnection,
    table_key: str,
) -> PreparedStatements:
    """Return prepared SQL using registry-derived column order for a table.

    Parameters
    ----------
    _unused_con
        DuckDB connection (kept for backward compatibility; not used).
    table_key
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
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is not None:
        registry_cols = [col.name for col in schema.columns]
    else:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is not None and contract.schema is not None:
            registry_cols = [col.name for col in contract.schema.columns]
        else:
            message = f"Table {table_key} missing from TABLE_SCHEMAS"
            raise RuntimeError(message)

    insert_sql = build_insert_sql(table_key, registry_cols)
    table_sql = quote_table_key(table_key)
    select_sql = " ".join(("SELECT * FROM", table_sql, "WHERE repo = ? AND commit = ?"))

    select_params: list[object] | None = None
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
        select_sql=select_sql,
        select_params=select_params,
    )


def ensure_schema(con: DuckDBPyConnection, table_key: str) -> None:
    """Validate that the live DuckDB table matches the registry definition.

    This:
    - Ensures that the literal column lists in `ingestion_sql` haven't drifted
      from the registry (once per process).
    - Ensures that the DuckDB table's columns & order match the registry.

    Parameters
    ----------
    con
        Active DuckDB connection.
    table_key
        Fully qualified table name (schema.table).

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    schema = TABLE_SCHEMAS.get(table_key)
    if contract is not None and contract.schema is not None:
        registry_cols = [col.name for col in contract.schema.columns]
        is_view = contract.is_view
    elif schema is not None:
        registry_cols = [col.name for col in schema.columns]
        is_view = False
    else:
        message = f"Table {table_key} missing from TABLE_SCHEMAS"
        raise RuntimeError(message)
    if is_view:
        return

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
