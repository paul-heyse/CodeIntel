"""Ingest macro registration helpers (storage-only, no ingestion/runtime deps)."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

from codeintel.storage.metadata_bootstrap import (
    INGEST_MACRO_DDLS,
    INGEST_MACROS,
    METADATA_SCHEMA_DDL_BASE,
)

_MACRO_CACHE: dict[int, set[str]] = {}
__all__ = [
    "assert_ingest_macros_present",
    "clear_macro_cache_for_connection",
    "ensure_ingest_macros",
    "list_ingest_macros",
]


def _registered_macros(con: DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        """
        SELECT schema_name, function_name
        FROM duckdb_functions()
        WHERE function_type IN ('macro', 'table_macro')
        """
    ).fetchall()
    names: set[str] = set()
    for schema_name, function_name in rows:
        if isinstance(function_name, str):
            fn_lower = function_name.lower()
            names.add(fn_lower)
            if isinstance(schema_name, str) and schema_name:
                names.add(f"{schema_name.lower()}.{fn_lower}")
            # If function_name contains a catalog/schema prefix, keep schema.fn and fn variants.
            parts = fn_lower.split(".")
            if len(parts) > 1:
                schema_part, fn_part = parts[-2], parts[-1]
                names.add(f"{schema_part}.{fn_part}")
                names.add(fn_part)
    return names


def ensure_ingest_macros(con: DuckDBPyConnection) -> None:
    """
    Ensure all ingest macros are registered for the active connection.

    Idempotent and cached per-connection to avoid redundant work.

    Raises
    ------
    RuntimeError
        If macros cannot be registered on the connection.
    """
    cache_key = id(con)
    cached = _MACRO_CACHE.get(cache_key, set())
    macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
    if macro_set.issubset(cached):
        # Verify the macros actually exist; connection ids can be recycled after close.
        registered = _registered_macros(con)
        if macro_set.issubset(registered):
            return
        _MACRO_CACHE.pop(cache_key, None)
    con.execute("\n".join(METADATA_SCHEMA_DDL_BASE))
    for ddl in INGEST_MACRO_DDLS:
        con.execute(ddl)

    registered = _registered_macros(con)
    if not macro_set.issubset(registered):
        # Retry once to account for transient creation issues.
        for ddl in INGEST_MACRO_DDLS:
            con.execute(ddl)
        registered = _registered_macros(con)
    if not macro_set.issubset(registered):
        missing = sorted(macro_set.difference(registered))
        message = f"Ingest macros missing after registration: {missing}"
        raise RuntimeError(message)

    updated = registered if registered else macro_set
    _MACRO_CACHE[cache_key] = updated


def clear_macro_cache_for_connection(con_or_key: DuckDBPyConnection | int) -> None:
    """Clear cached macro names for the given connection id or object."""
    cache_key = con_or_key if isinstance(con_or_key, int) else id(con_or_key)
    _MACRO_CACHE.pop(cache_key, None)


def list_ingest_macros(con: DuckDBPyConnection) -> set[str]:
    """
    Return registered ingest macro names (qualified and unqualified).

    Returns
    -------
    set[str]
        Macro names visible to the active connection.
    """
    return _registered_macros(con)


def assert_ingest_macros_present(con: DuckDBPyConnection) -> None:
    """
    Raise RuntimeError if any ingest macros are missing on the connection.

    Raises
    ------
    RuntimeError
        When one or more ingest macros are not visible to the connection.
    """
    macros = list_ingest_macros(con)
    macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
    if macro_set.issubset(macros):
        return
    # Retry a registration attempt before failing.
    for ddl in INGEST_MACRO_DDLS:
        con.execute(ddl)
    macros = list_ingest_macros(con)
    if macro_set.issubset(macros):
        return
    missing = sorted(macro_set.difference(macros))
    message = f"Ingest macros missing on connection: {missing}"
    raise RuntimeError(message)
