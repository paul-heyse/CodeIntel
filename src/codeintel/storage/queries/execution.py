"""Low-level DuckDB query execution helpers.

This module centralizes a tiny surface of direct DuckDB SQL execution for
application layers that otherwise operate on higher-level objects (Ibis
expressions, typed repositories, etc.). Application code should not call
``con.execute`` directly; instead it should route through helpers in the storage
layer like those defined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = [
    "duckdb_schema_exists",
    "duckdb_table_exists",
    "execute_sql",
]


def execute_sql(
    con: DuckDBConnection,
    sql: str,
    params: Sequence[object] | None = None,
) -> DuckDBConnection:
    """Execute a parameterized SQL statement.

    Parameters
    ----------
    con
        DuckDB connection to execute against.
    sql
        SQL string using DuckDB dialect, optionally containing positional parameters (``?``).
    params
        Optional parameter values bound to positional markers in ``sql``.

    Returns
    -------
    DuckDBConnection
        The same connection, enabling chained ``fetchone``/``df``/``pl`` calls.
    """
    if params is None:
        return con.execute(sql)
    return con.execute(sql, params)


def duckdb_table_exists(con: DuckDBConnection, *, schema: str, table: str) -> bool:
    """Return True when a DuckDB table exists.

    Parameters
    ----------
    con
        DuckDB connection.
    schema
        Schema name (e.g., ``"analytics"``).
    table
        Table name (e.g., ``"function_metrics"``).

    Returns
    -------
    bool
        True if the table is present in ``information_schema.tables``.
    """
    row = execute_sql(
        con,
        "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
        [schema, table],
    ).fetchone()
    return row is not None


def duckdb_schema_exists(con: DuckDBConnection, *, schema: str) -> bool:
    """Return True when a DuckDB schema exists.

    Parameters
    ----------
    con
        DuckDB connection.
    schema
        Schema name to check.

    Returns
    -------
    bool
        True if the schema is present in ``information_schema.schemata``.
    """
    row = execute_sql(
        con,
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ? LIMIT 1",
        [schema],
    ).fetchone()
    return row is not None
