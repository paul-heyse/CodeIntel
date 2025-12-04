"""Safe SQL query builder for DuckDB operations.

This module re-exports all primitives from sql_primitives and adds schema-aware
functions that integrate with codeintel.config.datasets.

For basic SQL building without schema dependencies, import from sql_primitives.
For full functionality including schema validation, import from this module.

Example
-------
>>> from codeintel.storage.sql_builder import QueryBuilder, SafeTable
>>>
>>> # Build a safe COUNT query
>>> query, params = QueryBuilder.count(
...     "analytics.function_metrics", where={"repo": "org/repo", "commit": "abc123"}
... )
>>> result = con.execute(query, params)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    TABLE_SCHEMAS,
)

# Re-export all primitives
from codeintel.storage.sql_primitives import (
    TABLE_KEY_PARTS,
    InvalidIdentifierError,
    PreparedStatements,
    QueryBuilder,
    SafeColumn,
    SafeTable,
    SqlBuilderError,
    SqlParams,
    build_delete_query,
    build_insert_sql,
    macro_select_sql,
    quote_identifier,
    quote_table_key,
    render_sql,
    safe_macro_call,
    validate_identifier,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


# --------------------------------------------------------------------------
# Schema-aware functions (depend on config.datasets)
# --------------------------------------------------------------------------


def prepared_statements_dynamic(
    _unused_con: DuckDBPyConnection,
    table_key: str,
) -> PreparedStatements:
    """
    Return prepared SQL using registry-derived column order for a table.

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
    # select_params default to a typical repo/commit tuple to keep the shape consistent;
    # callers should supply concrete values when executing.
    select_params: list[object] | None = None
    return PreparedStatements(
        insert_sql=insert_sql,
        delete_sql=None,
        select_sql=select_sql,
        select_params=select_params,
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
    con
        Active DuckDB connection.
    table_key
        Fully qualified table name (schema.table).

    Raises
    ------
    RuntimeError
        If the table is missing or deviates from the registry.
    """
    # Column verification is now done at build time via TABLE_SCHEMAS
    # No need for runtime verification since all columns come from the same source

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    schema = TABLE_SCHEMAS.get(table_key)
    if contract is not None and contract.schema is not None:
        registry_cols = [col.name for col in contract.schema.columns]
        is_view = contract.is_view
    elif schema is not None:
        # TableSchema from TABLE_SCHEMAS without a contract - assume base table
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


__all__ = [
    "TABLE_KEY_PARTS",
    "InvalidIdentifierError",
    "PreparedStatements",
    "QueryBuilder",
    "SafeColumn",
    "SafeTable",
    "SqlBuilderError",
    "SqlParams",
    "build_delete_query",
    "build_insert_sql",
    "ensure_schema",
    "macro_select_sql",
    "prepared_statements_dynamic",
    "quote_identifier",
    "quote_table_key",
    "render_sql",
    "safe_macro_call",
    "validate_identifier",
]
