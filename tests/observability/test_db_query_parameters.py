"""Tests for db.query.parameter emission helpers."""

from __future__ import annotations

from typing import cast

from codeintel.observability.db_tracing import DbQueryParameterConfig, emit_db_query_parameters
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)


def test_parameters_disabled_by_default() -> None:
    """Verify disabled configuration emits no attributes."""
    config = DbQueryParameterConfig(enabled=False, allowed_keys=frozenset({"limit"}))
    attrs = emit_db_query_parameters(
        sql="SELECT * FROM t WHERE x = $limit",
        params={"limit": 10},
        db_system_name="duckdb",
        config=config,
    )
    expect_equal(attrs, {})


def test_parameters_require_mapping() -> None:
    """Require a mapping input for parameter extraction."""
    config = DbQueryParameterConfig(enabled=True, allowed_keys=frozenset({"limit"}))
    attrs = emit_db_query_parameters(
        sql="SELECT * FROM t WHERE x = $limit",
        params=[10],
        db_system_name="duckdb",
        config=config,
    )
    expect_equal(attrs, {})


def test_parameters_allowlist_and_in_sql_gate() -> None:
    """Restrict emission to allowlisted keys present in SQL."""
    config = DbQueryParameterConfig(
        enabled=True,
        allowed_keys=frozenset({"limit", "offset"}),
        require_key_in_sql=True,
    )
    attrs = emit_db_query_parameters(
        sql="SELECT * FROM t LIMIT $limit",
        params={"limit": 25, "offset": 100},
        db_system_name="duckdb",
        config=config,
    )
    expect_equal(attrs, {"db.query.parameter.limit": 25})


def test_parameters_truncate_strings() -> None:
    """Truncate long string parameters to the configured length."""
    config = DbQueryParameterConfig(
        enabled=True,
        allowed_keys=frozenset({"q"}),
        max_string_len=5,
        require_key_in_sql=True,
    )
    attrs = emit_db_query_parameters(
        sql="SELECT * FROM t WHERE q = $q",
        params={"q": "abcdefgh"},
        db_system_name="duckdb",
        config=config,
    )
    value = attrs["db.query.parameter.q"]
    expect_is_instance(value, str)
    expect_true(
        cast("str", value).startswith("ab"),
        message="expected truncated parameter value",
    )


def test_parameters_skip_batches() -> None:
    """Skip parameter emission for batch operations."""
    config = DbQueryParameterConfig(enabled=True, allowed_keys=frozenset({"limit"}))
    attrs = emit_db_query_parameters(
        sql="INSERT INTO t VALUES ($limit)",
        params={"limit": 1},
        db_system_name="duckdb",
        config=config,
        is_batch=True,
    )
    expect_equal(attrs, {})
