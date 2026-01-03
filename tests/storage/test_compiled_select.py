"""Tests for SQLGlot-based SQL helpers."""

from __future__ import annotations

from codeintel.core.sqlglot_tools import extract_table_keys_duckdb, fingerprint_sql_duckdb
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_sqlglot_tools_extracts_table_keys_and_fingerprint() -> None:
    """SQLGlot helpers derive table keys and stable fingerprints."""
    sql = "  select * from core.modules  "
    table_keys = extract_table_keys_duckdb(sql)
    expect_true("core.modules" in table_keys, message="table key extracted")
    expect_equal(
        fingerprint_sql_duckdb(sql), fingerprint_sql_duckdb(sql.strip()), label="fingerprint"
    )
