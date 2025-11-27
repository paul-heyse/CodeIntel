"""Validate ingestion SQL registry against DuckDB information_schema."""

from __future__ import annotations

import pytest

from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.storage.gateway import DuckDBConnection, StorageGateway


def _table_exists(con: DuckDBConnection, schema_name: str, table_name: str) -> bool:
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


def _columns_for(con: DuckDBConnection, schema_name: str, table_name: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema_name, table_name],
    ).fetchall()
    return {row[0] for row in rows}


def test_ingestion_sql_tables_match_schema(fresh_gateway: StorageGateway) -> None:
    """Ensure registry tables exist with expected columns."""
    con = fresh_gateway.con
    registry = load_registry_columns(con)
    for fq_name, registry_cols in sorted(registry.items()):
        schema_name, table_name = fq_name.split(".", maxsplit=1)
        if not _table_exists(con, schema_name, table_name):
            pytest.fail(f"{table_name} table is missing")
        present = _columns_for(con, schema_name, table_name)
        expected = set(registry_cols)
        if not expected.issubset(present):
            missing = expected - present
            pytest.fail(f"{table_name} columns drift: missing {missing}")
