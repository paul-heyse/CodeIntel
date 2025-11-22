"""Validate ingestion SQL registry against DuckDB information_schema."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.config.schemas.tables import TABLE_SCHEMAS


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return result is not None


def _columns_for(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {row[0] for row in rows}


@pytest.mark.parametrize("fq_name", sorted(TABLE_SCHEMAS.keys()))
def test_ingestion_sql_tables_match_schema(fq_name: str) -> None:
    """Ensure registry tables exist with expected columns."""
    schema = TABLE_SCHEMAS[fq_name]
    con = duckdb.connect(database=":memory:")
    con.execute("CREATE SCHEMA IF NOT EXISTS core")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    table_name = schema.name
    cols = schema.column_names()
    con.execute(f"CREATE TABLE {table_name} ({', '.join(col + ' TEXT' for col in cols)})")

    if not _table_exists(con, table_name):
        pytest.fail(f"{table_name} table is missing")
    present = _columns_for(con, table_name)
    expected = set(cols)
    if not expected.issubset(present):
        missing = expected - present
        pytest.fail(f"{table_name} columns drift: missing {missing}")
