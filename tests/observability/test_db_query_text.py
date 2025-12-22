"""Tests for db.query.text helpers."""

from __future__ import annotations

from typing import cast

from codeintel.observability.db_query_text import (
    DbQueryTextConfig,
    DbQueryTextPolicy,
    looks_parameterized,
    redact_sql_literals_with_sqlglot,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_is_instance,
    expect_not_in,
    expect_true,
)


def test_looks_parameterized_detects_duckdb_placeholders() -> None:
    """Detect DuckDB placeholders in SQL text."""
    sql = "SELECT * FROM t WHERE id = $id AND name = ?"
    expect_true(looks_parameterized(sql, db_system_name="duckdb"))


def test_redact_sql_literals_replaces_literals() -> None:
    """Redact SQL literal values using sqlglot transformation."""
    config = DbQueryTextConfig(policy=DbQueryTextPolicy.REDACTED, max_len=200)
    sql = "SELECT * FROM users WHERE email = 'a@b.com' AND age > 42"
    redacted = redact_sql_literals_with_sqlglot(sql, dialect="duckdb", config=config)
    expect_is_instance(redacted, str)
    redacted_text = cast("str", redacted)
    expect_not_in("a@b.com", redacted_text)
    expect_not_in("42", redacted_text)
    expect_in("?", redacted_text)
