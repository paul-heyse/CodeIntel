"""Shared helpers for isolated gateway/DuckDB test setup."""

from __future__ import annotations

from pathlib import Path

import duckdb


def open_fresh_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Return a fresh DuckDB connection for tests.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Open connection (caller must close).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def seed_tables(con: duckdb.DuckDBPyConnection, ddl: list[str]) -> None:
    """Apply defensive DDL statements (DROP/CREATE) to avoid cross-test conflicts."""
    for stmt in ddl:
        con.execute(stmt)
