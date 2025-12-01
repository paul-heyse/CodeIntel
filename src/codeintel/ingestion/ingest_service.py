"""Centralized DuckDB ingestion helpers (registry- and macro-aware)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from duckdb import Error as DuckDBError

from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.ingest_macros import (
    assert_ingest_macros_present,
    ensure_ingest_macros,
    list_ingest_macros,
)
from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.sql_helpers import ensure_schema as _ensure_schema

log = logging.getLogger(__name__)
SMALL_BATCH_THRESHOLD = 25

if TYPE_CHECKING:
    from codeintel.storage.gateway import DuckDBConnection
else:
    DuckDBConnection = Any

INGEST_MACROS: dict[str, str] = {
    table_key: f"metadata.ingest_{table_key.split('.', maxsplit=1)[1]}"
    for table_key in TABLE_SCHEMAS
    if not table_key.startswith("metadata.")
}
INGEST_MACRO_TABLES: set[str] = set(INGEST_MACROS)
_MACRO_CACHE: dict[int, set[str]] = {}


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    _ensure_schema(con, table_key)


def _quote_identifier(identifier: str) -> str:
    """
    Return a safely quoted SQL identifier or raise when invalid.

    Returns
    -------
    str
        Identifier wrapped in double quotes.

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
    """
    Quote schema and table components after validating against the registry.

    Returns
    -------
    tuple[str, str, str]
        Schema name, table name, and fully quoted identifier.

    Raises
    ------
    ValueError
        If the table is not present in TABLE_SCHEMAS or components are unsafe.
    """
    if table_key not in TABLE_SCHEMAS:
        message = f"Unknown table key: {table_key}"
        raise ValueError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    quoted = f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"
    return schema_name, table_name, quoted


def _quote_macro_name(macro_name: str) -> str:
    """
    Return a validated macro identifier (optionally schema-qualified).

    Returns
    -------
    str
        Validated macro name with original casing preserved.

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


def _assert_macro_available(con: DuckDBConnection, macro_name: str) -> bool:
    """
    Ensure a specific ingest macro is available on the connection.

    Returns
    -------
    bool
        True when the macro is present, False when absent after retries.
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


def _prepare_registry(con: DuckDBConnection, table_key: str) -> tuple[list[str], str]:
    """
    Ensure schemas/macros exist and return registry columns plus macro name.

    Returns
    -------
    tuple[list[str], str]
        Registry column order and resolved macro name.

    Raises
    ------
    RuntimeError
        When registry metadata is missing.
    """
    ensure_ingest_macros(con)
    assert_ingest_macros_present(con)
    try:
        ensure_schema(con, table_key)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "missing" not in message:
            raise
        apply_all_schemas(con)
        ensure_schema(con, table_key)
    registry_cols = load_registry_columns(con).get(table_key)
    if registry_cols is None:
        message = f"Table {table_key} missing from registry"
        raise RuntimeError(message)
    macro_name = INGEST_MACROS.get(table_key)
    if macro_name is None:
        message = f"No ingest macro is defined for table {table_key}"
        raise RuntimeError(message)
    return registry_cols, macro_name


def _extract_repo_commit(
    registry_cols: Sequence[str], rows: Sequence[Sequence[object]]
) -> tuple[object | None, object | None]:
    """
    Pull repo/commit values from the first row for logging.

    Returns
    -------
    tuple[object | None, object | None]
        Repo value and commit value if present in the first row.
    """
    repo_val: object | None = None
    commit_val: object | None = None
    if not rows:
        return repo_val, commit_val
    try:
        repo_idx = registry_cols.index("repo")
        repo_val = rows[0][repo_idx]
    except ValueError:
        repo_val = None
    try:
        commit_idx = registry_cols.index("commit")
        commit_val = rows[0][commit_idx]
    except ValueError:
        commit_val = None
    return repo_val, commit_val


def _fallback_prepared_insert(
    con: DuckDBConnection,
    *,
    table_key: str,
    registry_cols: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> int:
    """
    Fallback path when macros are unavailable.

    Returns
    -------
    int
        Number of rows inserted via prepared append.
    """
    _, _, table_sql = _quote_table_key(table_key)
    df = pd.DataFrame([tuple(row) for row in rows], columns=pd.Index(registry_cols))
    try:
        con.from_df(df).insert_into(table_sql)
    except DuckDBError:
        # Table may not exist; apply schemas and retry
        apply_all_schemas(con)
        con.from_df(df).insert_into(table_sql)
    return len(rows)


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
    ValueError
        If table or macro identifiers are unsafe.
    """
    if not rows:
        return 0
    registry_cols, macro_name = _prepare_registry(con, table_key)
    # For small batches, the overhead of macro invocation outweighs benefits; use prepared insert.
    if len(rows) <= SMALL_BATCH_THRESHOLD:
        return _fallback_prepared_insert(
            con,
            table_key=table_key,
            registry_cols=registry_cols,
            rows=rows,
        )
    if not _assert_macro_available(con, macro_name):
        # Macros unavailable - ensure schema exists before fallback
        apply_all_schemas(con)
        repo_val, commit_val = _extract_repo_commit(registry_cols, rows)
        log.warning(
            "Falling back to prepared insert; macro missing",
            extra={
                "table_key": table_key,
                "macro_name": macro_name,
                "repo": repo_val,
                "commit": commit_val,
            },
        )
        try:
            return _fallback_prepared_insert(
                con, table_key=table_key, registry_cols=registry_cols, rows=rows
            )
        except DuckDBError:
            apply_all_schemas(con)
            return _fallback_prepared_insert(
                con, table_key=table_key, registry_cols=registry_cols, rows=rows
            )
        except ValueError as exc:
            message = f"Unsafe identifiers for ingest table {table_key}"
            raise ValueError(message) from exc

    try:
        _, _, table_sql = _quote_table_key(table_key)
        safe_macro = _quote_macro_name(macro_name)
    except ValueError as exc:
        message = f"Unsafe identifiers for ingest table {table_key}"
        raise ValueError(message) from exc

    schema_rel = con.table(table_sql)
    con.execute("DROP TABLE IF EXISTS temp_ingest_values")
    schema_rel.limit(0).create("temp_ingest_values")
    df = pd.DataFrame([tuple(row) for row in rows], columns=pd.Index(registry_cols))
    con.append("temp_ingest_values", df, by_name=True)

    try:
        # Use relation-based insert to avoid direct string interpolation in execute
        macro_rel = con.sql(
            "".join(
                ("SELECT * FROM ", safe_macro, "('temp_ingest_values')"),
            )
        )
        macro_rel.insert_into(table_sql)
        return len(rows)
    except DuckDBError:
        log.warning(
            "Macro unavailable at execution time; falling back to prepared insert",
            extra={"table_key": table_key, "macro_name": macro_name},
            exc_info=True,
        )
        return _fallback_prepared_insert(
            con,
            table_key=table_key,
            registry_cols=registry_cols,
            rows=rows,
        )


__all__ = [
    "INGEST_MACRO_TABLES",
    "ensure_schema",
    "ingest_via_macro",
    "macro_exists",
]
