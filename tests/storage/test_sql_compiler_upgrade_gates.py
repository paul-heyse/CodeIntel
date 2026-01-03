"""Compiler upgrade gates for storage schema and DDL rendering.

These tests snapshot the canonical SQL output of the project's schema round-trip
layer (TableSchema → Ibis schema → SQLGlot AST → DuckDB SQL). They help detect
upstream library upgrades that alter type rendering or nullability semantics.
"""

from __future__ import annotations

import duckdb
from sqlglot import parse_one

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.sqlglot_tools import semantic_diff_sql_duckdb
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.schema_roundtrip import create_table_ast
from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.schemas import ensure_production_schemas

_MIN_EXPECTED_COLUMNS = 2


def _canonical_duckdb_sql(sql: str) -> str:
    """Return canonical DuckDB SQL text for stable golden comparisons.

    Parameters
    ----------
    sql
        DuckDB SQL text to normalize.

    Returns
    -------
    str
        Canonicalized SQL string.
    """
    return parse_one(sql, dialect=DUCKDB_DIALECT).sql(dialect=DUCKDB_DIALECT)


def test_compiler_upgrade_gate_create_table_sql_is_stable() -> None:
    """Canonical CREATE TABLE SQL remains stable for representative types."""
    schema = TableSchema(
        schema="core",
        name="demo_types",
        columns=[
            Column(name="id", type="BIGINT", nullable=False),
            Column(name="flag", type="BOOLEAN", nullable=False),
            Column(name="count", type="INTEGER", nullable=True),
            Column(name="ratio", type="DOUBLE", nullable=True),
            Column(name="name", type="VARCHAR", nullable=False),
            Column(name="payload", type="JSON", nullable=True),
            Column(name="ts", type="TIMESTAMP", nullable=True),
            Column(name="ts_tz", type="TIMESTAMPTZ", nullable=True),
            Column(name="amount", type="DECIMAL(38,0)", nullable=True),
        ],
        primary_key=("id",),
    )

    sql = create_table_ast(schema, if_not_exists=True).sql(dialect=DUCKDB_DIALECT)
    canonical = _canonical_duckdb_sql(sql)
    expected = (
        "CREATE TABLE IF NOT EXISTS core.demo_types (id BIGINT NOT NULL, "
        "flag BOOLEAN NOT NULL, count INT, ratio DOUBLE, name TEXT NOT NULL, "
        "payload JSON, ts TIMESTAMP, ts_tz TIMESTAMPTZ, amount DECIMAL(38, 0), "
        "PRIMARY KEY (id))"
    )
    if canonical != expected:
        diff = semantic_diff_sql_duckdb(expected, canonical)
        message = f"Canonical SQL mismatch. Semantic diff: {diff}"
        expect_true(condition=False, message=message)


def test_compiler_upgrade_gate_create_table_executes() -> None:
    """Rendered DDL executes successfully on DuckDB."""
    schema = TableSchema(
        schema="core",
        name="demo_types",
        columns=[
            Column(name="id", type="BIGINT", nullable=False),
            Column(name="flag", type="BOOLEAN", nullable=False),
        ],
        primary_key=("id",),
    )
    sql = create_table_ast(schema, if_not_exists=True).sql(dialect=DUCKDB_DIALECT)

    con = duckdb.connect()
    ensure_production_schemas(con)
    con.execute(sql)
    rows = con.execute("PRAGMA table_info('core.demo_types')").fetchall()
    con.close()

    expect_true(
        len(rows) >= _MIN_EXPECTED_COLUMNS,
        message="Expected PRAGMA table_info to return at least 2 columns",
    )
