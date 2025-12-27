"""DuckDB catalog helpers shared across storage modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = ["duckdb_default_catalog", "duckdb_schema_exists"]


def duckdb_default_catalog(con: DuckDBPyConnection) -> str | None:
    """Return the primary catalog name for a DuckDB connection.

    Parameters
    ----------
    con
        DuckDB connection to query.

    Returns
    -------
    str | None
        Primary catalog name, or None when unavailable.
    """
    row = con.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    catalog = row[1]
    if isinstance(catalog, str) and catalog.strip():
        return catalog
    return None


def duckdb_schema_exists(con: DuckDBPyConnection, *, schema: str) -> bool:
    """Return True when a DuckDB schema exists.

    Parameters
    ----------
    con
        DuckDB connection to query.
    schema
        Schema name to check.

    Returns
    -------
    bool
        True when the schema exists.
    """
    row = con.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
        [schema],
    ).fetchone()
    return row is not None
