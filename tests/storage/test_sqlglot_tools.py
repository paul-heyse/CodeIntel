"""Tests for storage SQLGlot toolkit primitives."""

from __future__ import annotations

import pytest

from codeintel.storage.sqlglot_tools import (
    ParseError,
    canonical_sql_duckdb,
    extract_column_lineage_duckdb,
    extract_table_keys_duckdb,
    fingerprint_sql_duckdb,
    parse_one_duckdb,
)
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_parse_one_duckdb_parses_valid_sql() -> None:
    """parse_one_duckdb returns an AST for valid SQL."""
    root = parse_one_duckdb("SELECT 1 AS x")
    expect_true(root is not None, message="parsed AST is present")


def test_parse_one_duckdb_raises_parse_error_on_invalid_sql() -> None:
    """parse_one_duckdb raises ParseError on invalid SQL."""
    with pytest.raises(ParseError):
        parse_one_duckdb("SELECT FROM")


def test_extract_table_keys_is_cte_safe() -> None:
    """extract_table_keys_duckdb ignores CTE names as physical tables."""
    sql = """
    WITH t AS (
        SELECT 1 AS x
    )
    SELECT *
    FROM t
    JOIN core.modules m ON 1 = 1
    """
    keys = extract_table_keys_duckdb(sql)
    expect_true("core.modules" in keys, message="physical table present")
    expect_true("t" not in keys, message="CTE name not treated as table ref")


def test_extract_table_keys_handles_nested_subqueries() -> None:
    """extract_table_keys_duckdb finds tables in nested subqueries."""
    sql = "SELECT * FROM (SELECT * FROM analytics.function_metrics) sub"
    keys = extract_table_keys_duckdb(sql)
    expect_equal(keys, frozenset({"analytics.function_metrics"}))


def test_fingerprint_is_stable_for_equivalent_sql() -> None:
    """fingerprint_sql_duckdb is stable for whitespace/casing variants."""
    a = fingerprint_sql_duckdb("SELECT 1")
    b = fingerprint_sql_duckdb("  select 1  ")
    expect_equal(a, b, label="fingerprint")


def test_canonical_sql_is_stable_for_equivalent_sql() -> None:
    """canonical_sql_duckdb produces stable rendering for equivalent SQL."""
    a = canonical_sql_duckdb("SELECT 1")
    b = canonical_sql_duckdb("  select 1  ")
    expect_equal(a, b, label="canonical")


def test_extract_column_lineage_maps_output_columns() -> None:
    """extract_column_lineage_duckdb returns upstream column references."""
    sql = """
    SELECT
        m.module AS module,
        m.repo || ':' || m.commit AS repo_commit
    FROM core.modules AS m
    """
    lineage = extract_column_lineage_duckdb(sql)
    expect_equal(lineage["module"], frozenset({"core.modules.module"}))
    expect_equal(
        lineage["repo_commit"],
        frozenset({"core.modules.repo", "core.modules.commit"}),
    )
