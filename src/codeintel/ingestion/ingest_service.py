"""Centralized DuckDB ingestion helpers (registry- and macro-aware)."""

# ruff: noqa: S608  # registry-controlled identifiers are injected into SQL strings
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from codeintel.config.schemas.ingestion_sql import verify_ingestion_columns
from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.config.schemas.sql_builder import ensure_schema as _ensure_schema
from codeintel.config.schemas.tables import TABLE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection
else:
    DuckDBConnection = Any

_VALIDATED: list[bool] = [False]
INGEST_MACROS: dict[str, str] = {
    table_key: f"metadata.ingest_{table_key.split('.', maxsplit=1)[1]}"
    for table_key in TABLE_SCHEMAS
    if not table_key.startswith("metadata.")
}
INGEST_MACRO_TABLES: set[str] = set(INGEST_MACROS)
_MACRO_CACHE: dict[int, set[str]] = {}


def _load_macro_names(con: DuckDBConnection) -> set[str]:
    """
    Return macro names (qualified + unqualified) for the active connection.

    Returns
    -------
    set[str]
        Qualified and unqualified macro names for the connection.
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


def ensure_schema(con: DuckDBConnection, table_key: str) -> None:
    """Validate registry alignment once per process and ensure the table matches the registry."""
    if not _VALIDATED[0]:
        verify_ingestion_columns(con)
        _VALIDATED[0] = True
    _ensure_schema(con, table_key)


def macro_exists(con: DuckDBConnection, macro_name: str) -> bool:
    """
    Return True when a macro is registered.

    Returns
    -------
    bool
        True if the macro is present, otherwise False.
    """
    cache_key = id(con)
    names = _MACRO_CACHE.get(cache_key)
    if names is None:
        names = _load_macro_names(con)
        _MACRO_CACHE[cache_key] = names
    target = macro_name.lower()
    short = target.split(".", maxsplit=1)[-1]
    if target in names or short in names:
        return True
    # Refresh once in case macros were just created.
    _MACRO_CACHE.pop(cache_key, None)
    refreshed = _load_macro_names(con)
    _MACRO_CACHE[cache_key] = refreshed
    return target in refreshed or short in refreshed


def ingest_via_macro(
    con: DuckDBConnection,
    table_key: str,
    rows: Sequence[Sequence[object]],
) -> int:
    """
    Insert rows using a macro-backed path when available; otherwise fall back to prepared inserts.

    Parameters
    ----------
    con :
        Active DuckDB connection.
    table_key :
        Registry key (e.g., ``analytics.function_metrics``).
    rows :
        Row payloads matching the registry column order.

    Returns
    -------
    int
        Number of rows inserted.

    Raises
    ------
    RuntimeError
        If registry metadata for the table is missing.
    """
    if not rows:
        return 0
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)
    macro_name = INGEST_MACROS.get(table_key)
    if macro_name is None:
        message = f"No ingest macro is defined for table {table_key}"
        raise RuntimeError(message)
    if not macro_exists(con, macro_name):
        message = f"Ingest macro {macro_name} missing for table {table_key}"
        raise RuntimeError(message)

    placeholders = ", ".join("?" for _ in registry_cols)
    column_list = ", ".join(registry_cols)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'
    con.execute("DROP TABLE IF EXISTS temp_ingest_values")
    con.execute(f"CREATE TEMP TABLE temp_ingest_values AS SELECT * FROM {table_sql} WHERE 0=1")
    con.executemany(f"INSERT INTO temp_ingest_values ({column_list}) VALUES ({placeholders})", rows)
    con.execute(
        f"INSERT INTO {table_sql} SELECT * FROM {macro_name}(?)",
        ["temp_ingest_values"],
    )
    return len(rows)


__all__ = [
    "INGEST_MACRO_TABLES",
    "ensure_schema",
    "ingest_via_macro",
    "macro_exists",
]
