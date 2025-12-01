"""Connection-aware schema helpers driven by the DuckDB registry metadata."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import LiteralString, cast

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns

_INGESTION_COLUMNS_VERIFIED: list[bool] = [False]


@dataclass(frozen=True)
class PreparedStatements:
    """Prepared insert/delete/select SQL for a table (registry-driven)."""

    insert_sql: str
    delete_sql: str | None = None
    select_sql: str | None = None
    select_params: list[object] | None = None


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
TABLE_KEY_PARTS = 2


def quote_identifier(identifier: str) -> str:
    """
    Validate and quote a SQL identifier.

    Returns
    -------
    str
        Quoted identifier.

    Raises
    ------
    ValueError
        When the identifier is unsafe.
    """
    if not _IDENTIFIER_RE.fullmatch(identifier):
        message = f"Unsafe identifier: {identifier}"
        raise ValueError(message)
    return f'"{identifier}"'


def quote_table_key(table_key: str) -> str:
    """
    Validate and quote a fully qualified table key (schema.table).

    Returns
    -------
    str
        Quoted table identifier.

    Raises
    ------
    ValueError
        When the key is invalid.
    """
    parts = table_key.split(".")
    if len(parts) != TABLE_KEY_PARTS or not parts[0] or not parts[1]:
        message = f"Table key must include schema: {table_key}"
        raise ValueError(message)
    schema_name, table_name = parts
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def macro_select_sql(macro_name: str, placeholders: str) -> str:
    """
    Build a validated SELECT statement invoking a macro.

    Parameters
    ----------
    macro_name
        Fully qualified macro name (schema.macro).
    placeholders
        Placeholder string (e.g., "?, ?, ?").

    Returns
    -------
    str
        Safe SELECT statement invoking the macro.

    Raises
    ------
    ValueError
        If the macro name is unsafe.
    """
    if "." not in macro_name:
        message = f"Macro name must include schema: {macro_name}"
        raise ValueError(message)
    schema_name, macro = macro_name.split(".", maxsplit=1)
    macro_sql = ".".join((quote_identifier(schema_name), quote_identifier(macro)))
    query = "".join(
        (
            "SELECT * FROM /*",
            macro_name,
            "*/ ",
            macro_sql,
            "(",
            placeholders,
            ")",
        )
    )
    return cast("LiteralString", query)


def safe_macro_call(
    macro_name: str,
    args: Sequence[object],
    *,
    allowed: Collection[str] | None = None,
) -> tuple[str, Sequence[object]]:
    """
    Return a safe SELECT statement and args for a macro invocation.

    Parameters
    ----------
    macro_name
        Fully qualified macro name (schema.macro).
    args
        Parameters to pass to the macro.
    allowed
        Optional allowlist of macro names; when provided, macro_name must be present.

    Returns
    -------
    tuple[str, Sequence[object]]
        Parameterized SQL and the original args.

    Raises
    ------
    ValueError
        If the macro name is unsafe or not allowlisted.
    """
    if allowed is not None and macro_name not in allowed:
        message = f"Macro {macro_name} is not allowlisted"
        raise ValueError(message)
    placeholders = ", ".join("?" for _ in args)
    sql = macro_select_sql(macro_name, placeholders)
    return sql, args


def build_insert_sql(
    table_identifier: str,
    columns: Sequence[str],
    *,
    identifier_is_quoted: bool = False,
) -> str:
    """
    Build a parameterized INSERT statement with validated identifiers.

    Parameters
    ----------
    table_identifier
        Fully qualified table name (schema.table) or a pre-quoted identifier
        when ``identifier_is_quoted`` is True.
    columns
        Ordered list of column names to insert into.
    identifier_is_quoted
        When True, ``table_identifier`` is treated as already quoted (e.g., a
        temporary view name). When False, it is validated and quoted.

    Returns
    -------
    str
        INSERT statement with placeholders for values.
    """
    table_sql = table_identifier if identifier_is_quoted else quote_table_key(table_identifier)
    cols_sql = ", ".join(quote_identifier(col) for col in columns)
    placeholders = ", ".join("?" for _ in columns)
    query = "".join(
        (
            "INSERT INTO ",
            table_sql,
            " (",
            cols_sql,
            ") VALUES (",
            placeholders,
            ")",
        )
    )
    return cast("LiteralString", query)


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

    insert_sql = build_insert_sql(table_key, registry_cols)
    table_sql = quote_table_key(table_key)
    select_sql = cast(
        "LiteralString",
        " ".join(("SELECT * FROM", table_sql, "WHERE repo = ? AND commit = ?")),
    )
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


__all__ = [
    "PreparedStatements",
    "build_insert_sql",
    "ensure_schema",
    "macro_select_sql",
    "prepared_statements_dynamic",
    "quote_identifier",
    "quote_table_key",
    "safe_macro_call",
]
