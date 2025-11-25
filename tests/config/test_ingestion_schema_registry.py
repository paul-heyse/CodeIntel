"""Validate ingestion SQL registry against DuckDB information_schema."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import StorageGateway


def _table_exists(con: duckdb.DuckDBPyConnection, schema_name: str, table_name: str) -> bool:
    result = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        LIMIT 1
        """,
        [schema_name, table_name],
    ).fetchone()
    return result is not None


def _columns_for(con: duckdb.DuckDBPyConnection, schema_name: str, table_name: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema_name, table_name],
    ).fetchall()
    return {row[0] for row in rows}


@pytest.mark.parametrize("fq_name", sorted(TABLE_SCHEMAS.keys()))
def test_ingestion_sql_tables_match_schema(fq_name: str, fresh_gateway: StorageGateway) -> None:
    """Ensure registry tables exist with expected columns."""
    schema = TABLE_SCHEMAS[fq_name]
    con = fresh_gateway.con
    schema_name = schema.schema
    table_name = schema.name
    cols = schema.column_names()

    if not _table_exists(con, schema_name, table_name):
        pytest.fail(f"{table_name} table is missing")
    present = _columns_for(con, schema_name, table_name)
    expected = set(cols)
    if not expected.issubset(present):
        missing = expected - present
        pytest.fail(f"{table_name} columns drift: missing {missing}")
