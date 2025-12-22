"""Tests for db.query.text helpers."""

from __future__ import annotations

from codeintel.observability.db_query_text import (
    DbQueryTextConfig,
    DbQueryTextPolicy,
    looks_parameterized,
    redact_sql_literals_with_sqlglot,
)


def test_looks_parameterized_detects_duckdb_placeholders() -> None:
    """Detect DuckDB placeholders in SQL text."""
    sql = "SELECT * FROM t WHERE id = $id AND name = ?"
    assert looks_parameterized(sql, db_system_name="duckdb")


def test_redact_sql_literals_replaces_literals() -> None:
    """Redact SQL literal values using sqlglot transformation."""
    config = DbQueryTextConfig(policy=DbQueryTextPolicy.REDACTED, max_len=200)
    sql = "SELECT * FROM users WHERE email = 'a@b.com' AND age > 42"
    redacted = redact_sql_literals_with_sqlglot(sql, dialect="duckdb", config=config)
    assert redacted is not None
    assert "a@b.com" not in redacted
    assert "42" not in redacted
    assert "?" in redacted
