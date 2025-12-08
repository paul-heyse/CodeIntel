"""Comprehensive tests for SQL primitives module.

This module tests all SQL query building primitives in codeintel.storage.sql.primitives,
following the Testing Charter by using real DuckDB connections for validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.primitives import (
    InvalidIdentifierError,
    PreparedStatements,
    QueryBuilder,
    SafeColumn,
    SafeTable,
    SqlBuilderError,
    build_delete_query,
    build_insert_sql,
    macro_select_sql,
    quote_identifier,
    quote_table_key,
    render_sql,
    safe_macro_call,
    validate_identifier,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_in,
    expect_true,
    require_row,
)


def expect_query_contains(query: str, *parts: str) -> None:
    """Verify that each fragment appears in the SQL query."""
    for part in parts:
        expect_in(part, query)


def expect_query_not_contains(query: str, *parts: str) -> None:
    """Verify that each fragment is absent from the SQL query."""
    for part in parts:
        expect_not_in(part, query)


def expect_params(params: Sequence[object], expected: Sequence[object]) -> None:
    """Assert parameter lists match exactly."""
    expect_equal(params, expected, label="params")


# =============================================================================
# Exception Classes Tests
# =============================================================================


def test_sql_builder_error_inherits_from_exception() -> None:
    """Verify SqlBuilderError is a proper exception."""
    err = SqlBuilderError("test message")
    expect_is_instance(err, Exception)
    expect_equal(str(err), "test message")


def test_invalid_identifier_error_stores_attributes() -> None:
    """Verify error stores identifier and reason attributes."""
    err = InvalidIdentifierError("bad_id", "contains special chars")
    expect_equal(err.identifier, "bad_id")
    expect_equal(err.reason, "contains special chars")
    expect_in("bad_id", str(err))
    expect_in("contains special chars", str(err))


def test_invalid_identifier_error_inherits_from_sql_builder_error() -> None:
    """Verify it inherits from SqlBuilderError."""
    err = InvalidIdentifierError("test", "reason")
    expect_is_instance(err, SqlBuilderError)


# =============================================================================
# validate_identifier Tests
# =============================================================================


@pytest.mark.parametrize(
    "identifier",
    [
        "foo",
        "bar",
        "column_name",
        "Table1",
        "_private",
        "CamelCase",
        "snake_case_123",
        "A",
        "_",
        "schema.table",
        "core.modules",
        "analytics.function_metrics",
    ],
)
def test_validate_identifier_accepts_valid(identifier: str) -> None:
    """Verify valid identifiers pass validation."""
    result = validate_identifier(identifier)
    expect_equal(result, identifier)


def test_validate_identifier_rejects_empty() -> None:
    """Verify empty string raises InvalidIdentifierError."""
    with pytest.raises(InvalidIdentifierError, match="cannot be empty"):
        validate_identifier("")


def test_validate_identifier_rejects_overly_long() -> None:
    """Verify identifiers exceeding max length raise error."""
    long_id = "a" * 129  # 129 chars exceeds 128 limit
    with pytest.raises(InvalidIdentifierError, match="exceeds maximum length"):
        validate_identifier(long_id)


def test_validate_identifier_accepts_max_length() -> None:
    """Verify identifier at exactly max length is accepted."""
    max_id = "a" * 128
    result = validate_identifier(max_id)
    expect_equal(result, max_id)


@pytest.mark.parametrize(
    "identifier",
    [
        "1starts_with_number",
        "has-dash",
        "has space",
        "has;semicolon",
        "has'quote",
        'has"doublequote',
        "has.multiple.dots",
        "has..double.dots",
    ],
)
def test_validate_identifier_rejects_invalid_patterns(identifier: str) -> None:
    """Verify invalid patterns raise InvalidIdentifierError."""
    with pytest.raises(InvalidIdentifierError):
        validate_identifier(identifier)


# =============================================================================
# SafeTable Tests
# =============================================================================


def test_safe_table_creates_with_valid_name() -> None:
    """Verify SafeTable accepts valid table names."""
    table = SafeTable("analytics.metrics")
    expect_equal(table.name, "analytics.metrics")
    expect_equal(str(table), "analytics.metrics")


def test_safe_table_creates_with_simple_name() -> None:
    """Verify SafeTable accepts unqualified names."""
    table = SafeTable("modules")
    expect_equal(table.name, "modules")


def test_safe_table_raises_on_invalid_name() -> None:
    """Verify SafeTable raises on invalid names."""
    with pytest.raises(InvalidIdentifierError):
        SafeTable("1invalid")


def test_safe_table_raises_on_empty_name() -> None:
    """Verify SafeTable raises on empty name."""
    with pytest.raises(InvalidIdentifierError, match="cannot be empty"):
        SafeTable("")


def test_safe_table_is_frozen() -> None:
    """Verify SafeTable is immutable."""
    table = SafeTable("core.modules")
    assert_frozen(table, "name", "other")


# =============================================================================
# SafeColumn Tests
# =============================================================================


def test_safe_column_creates_with_valid_name() -> None:
    """Verify SafeColumn accepts valid column names."""
    col = SafeColumn("function_name")
    expect_equal(col.name, "function_name")
    expect_equal(str(col), "function_name")


@pytest.mark.parametrize(
    "name",
    ["id", "repo", "commit", "created_at", "goid_h128", "_private", "CamelCase"],
)
def test_safe_column_accepts_various_valid_names(name: str) -> None:
    """Verify various valid column names are accepted."""
    col = SafeColumn(name)
    expect_equal(col.name, name)


def test_safe_column_rejects_names_with_dots() -> None:
    """Verify column names with dots are rejected."""
    with pytest.raises(InvalidIdentifierError, match="cannot contain dots"):
        SafeColumn("schema.column")


def test_safe_column_rejects_invalid_names() -> None:
    """Verify invalid column names are rejected."""
    with pytest.raises(InvalidIdentifierError):
        SafeColumn("1invalid")


def test_safe_column_is_frozen() -> None:
    """Verify SafeColumn is immutable."""
    col = SafeColumn("repo")
    assert_frozen(col, "name", "other")


# =============================================================================
# QueryBuilder.count Tests
# =============================================================================


def test_query_builder_count_without_where() -> None:
    """Verify COUNT query without WHERE clause."""
    query, params = QueryBuilder.count("core.modules")
    expect_query_contains(query, "SELECT COUNT(*)", "core.modules")
    expect_query_not_contains(query, "WHERE")
    expect_params(params, [])


def test_query_builder_count_with_where() -> None:
    """Verify COUNT query with WHERE clause."""
    query, params = QueryBuilder.count(
        "core.modules",
        where={"repo": "test/repo", "commit": "abc123"},
    )
    expect_query_contains(query, "SELECT COUNT(*)", "WHERE", "repo = ?", "commit = ?")
    expect_params(params, ["test/repo", "abc123"])


def test_query_builder_count_with_safe_table() -> None:
    """Verify COUNT accepts SafeTable."""
    table = SafeTable("analytics.metrics")
    query, params = QueryBuilder.count(table)
    expect_query_contains(query, "analytics.metrics")
    expect_params(params, [])


def test_query_builder_count_executes_on_real_db(fresh_gateway: StorageGateway) -> None:
    """Verify COUNT query executes on real DuckDB."""
    query, params = QueryBuilder.count(
        "core.modules",
        where={"repo": "test/repo", "commit": "abc123"},
    )
    result = fresh_gateway.con.execute(query, params).fetchone()
    row = require_row(result, message="Expected COUNT(*) row")
    expect_equal(row[0], 0, label="row_count")


# =============================================================================
# QueryBuilder.count_where_null Tests
# =============================================================================


def test_query_builder_count_null_basic() -> None:
    """Verify COUNT with IS NULL condition."""
    query, _ = QueryBuilder.count_where_null("core.goids", "qualname")
    expect_query_contains(query, "SELECT COUNT(*)", "qualname IS NULL")


def test_query_builder_count_null_with_where() -> None:
    """Verify COUNT with IS NULL and additional WHERE conditions."""
    query, params = QueryBuilder.count_where_null(
        "core.goids",
        "qualname",
        where={"repo": "test/repo"},
    )
    expect_query_contains(query, "qualname IS NULL", "repo = ?")
    expect_params(params, ["test/repo"])


def test_query_builder_count_null_with_safe_column() -> None:
    """Verify COUNT accepts SafeColumn."""
    col = SafeColumn("description")
    query, _ = QueryBuilder.count_where_null("analytics.metrics", col)
    expect_query_contains(query, "description IS NULL")


# =============================================================================
# QueryBuilder.delete Tests
# =============================================================================


def test_query_builder_delete_with_single_condition() -> None:
    """Verify DELETE with single WHERE condition."""
    query, params = QueryBuilder.delete("core.modules", where={"repo": "test/repo"})
    expect_query_contains(query, "DELETE FROM", "core.modules", "WHERE", "repo = ?")
    expect_params(params, ["test/repo"])


def test_query_builder_delete_with_multiple_conditions() -> None:
    """Verify DELETE with multiple WHERE conditions."""
    query, params = QueryBuilder.delete(
        "core.modules",
        where={"repo": "test/repo", "commit": "abc123"},
    )
    expect_query_contains(query, "repo = ?", "commit = ?", "AND")
    expect_length(params, 2, label="param_count")


def test_query_builder_delete_with_safe_table() -> None:
    """Verify DELETE accepts SafeTable."""
    table = SafeTable("analytics.metrics")
    query, _ = QueryBuilder.delete(table, where={"id": 1})
    expect_query_contains(query, "analytics.metrics")


# =============================================================================
# QueryBuilder.delete_in Tests
# =============================================================================


def test_query_builder_delete_in_basic() -> None:
    """Verify DELETE with IN clause."""
    query, params = QueryBuilder.delete_in("core.modules", "repo", ["a", "b", "c"])
    expect_query_contains(query, "DELETE FROM", "IN (?, ?, ?)")
    expect_params(params, ["a", "b", "c"])


def test_query_builder_delete_in_single_value() -> None:
    """Verify DELETE IN with single value."""
    query, params = QueryBuilder.delete_in("core.modules", "id", [1])
    expect_query_contains(query, "IN (?)")
    expect_params(params, [1])


def test_query_builder_delete_in_with_safe_column() -> None:
    """Verify DELETE IN accepts SafeColumn."""
    col = SafeColumn("goid_h128")
    query, _ = QueryBuilder.delete_in("core.goids", col, [100, 200])
    expect_query_contains(query, "goid_h128 IN")


# =============================================================================
# QueryBuilder.select_all Tests
# =============================================================================


def test_query_builder_select_all_basic() -> None:
    """Verify SELECT * query."""
    query = QueryBuilder.select_all("core.modules")
    expect_equal(query, "SELECT * FROM core.modules")


def test_query_builder_select_all_with_safe_table() -> None:
    """Verify SELECT * accepts SafeTable."""
    table = SafeTable("analytics.metrics")
    query = QueryBuilder.select_all(table)
    expect_query_contains(query, "analytics.metrics")


def test_query_builder_select_all_executes_on_real_db(fresh_gateway: StorageGateway) -> None:
    """Verify SELECT * executes on real DuckDB."""
    query = QueryBuilder.select_all("core.modules")
    result = fresh_gateway.con.execute(query).fetchall()
    expect_is_instance(result, list)


# =============================================================================
# QueryBuilder.insert Tests
# =============================================================================


def test_query_builder_insert_basic() -> None:
    """Verify INSERT query generation."""
    query = QueryBuilder.insert("core.modules", ["module", "path", "repo", "commit"])
    expect_query_contains(query, "INSERT INTO", "core.modules", "module", "VALUES (?, ?, ?, ?)")


def test_query_builder_insert_single_column() -> None:
    """Verify INSERT with single column."""
    query = QueryBuilder.insert("test.table", ["col"])
    expect_query_contains(query, "VALUES (?)")


def test_query_builder_insert_with_safe_columns() -> None:
    """Verify INSERT accepts SafeColumn."""
    cols = [SafeColumn("repo"), SafeColumn("commit")]
    query = QueryBuilder.insert("core.modules", cols)
    expect_query_contains(query, "repo", "commit")


# =============================================================================
# QueryBuilder.delete_repo_commit Tests
# =============================================================================


def test_query_builder_delete_repo_commit_basic() -> None:
    """Verify standard repo/commit scoped DELETE."""
    query = QueryBuilder.delete_repo_commit("core.modules")
    expect_query_contains(query, "DELETE FROM", "repo = ?", "commit = ?", "AND")


def test_query_builder_delete_repo_commit_with_safe_table() -> None:
    """Verify delete_repo_commit accepts SafeTable."""
    table = SafeTable("analytics.metrics")
    query = QueryBuilder.delete_repo_commit(table)
    expect_query_contains(query, "analytics.metrics")


# =============================================================================
# build_delete_query Tests
# =============================================================================


def test_build_delete_query_returns_query_when_has_scope() -> None:
    """Verify returns DELETE query when has_scope is True."""
    query = build_delete_query("core.modules", has_scope=True)
    expect_is_not_none(query)
    expect_query_contains(query or "", "DELETE FROM", "repo = ?")


def test_build_delete_query_returns_none_when_no_scope() -> None:
    """Verify returns None when has_scope is False."""
    query = build_delete_query("core.modules", has_scope=False)
    expect_is_none(query)


# =============================================================================
# PreparedStatements Tests
# =============================================================================


def test_prepared_statements_creates_with_required_field() -> None:
    """Verify PreparedStatements with only insert_sql."""
    stmt = PreparedStatements(insert_sql="INSERT INTO test VALUES (?)")
    expect_equal(stmt.insert_sql, "INSERT INTO test VALUES (?)")
    expect_is_none(stmt.delete_sql)
    expect_is_none(stmt.select_sql)
    expect_is_none(stmt.select_params)


def test_prepared_statements_creates_with_all_fields() -> None:
    """Verify PreparedStatements with all fields."""
    stmt = PreparedStatements(
        insert_sql="INSERT INTO test VALUES (?)",
        delete_sql="DELETE FROM test WHERE id = ?",
        select_sql="SELECT * FROM test WHERE repo = ?",
        select_params=["test/repo"],
    )
    expect_is_not_none(stmt.insert_sql)
    expect_is_not_none(stmt.delete_sql)
    expect_is_not_none(stmt.select_sql)
    expect_equal(stmt.select_params, ["test/repo"])


def test_prepared_statements_is_frozen() -> None:
    """Verify PreparedStatements is immutable."""
    stmt = PreparedStatements(insert_sql="INSERT")
    assert_frozen(stmt, "insert_sql", "OTHER")


# =============================================================================
# render_sql Tests
# =============================================================================


def test_render_sql_joins_parts_with_spaces() -> None:
    """Verify parts are joined with spaces."""
    result = render_sql(["SELECT", "*", "FROM", "table"])
    expect_equal(result, "SELECT * FROM table")


def test_render_sql_filters_empty_parts() -> None:
    """Verify empty parts are filtered out."""
    result = render_sql(["SELECT", "", "*", "FROM", "table"])
    expect_equal(result, "SELECT * FROM table")


def test_render_sql_handles_single_part() -> None:
    """Verify single part is returned as-is."""
    result = render_sql(["SELECT *"])
    expect_equal(result, "SELECT *")


def test_render_sql_handles_empty_list() -> None:
    """Verify empty list returns empty string."""
    result = render_sql([])
    expect_true(not result, message="Expected empty string for empty parts")


# =============================================================================
# quote_identifier Tests
# =============================================================================


def test_quote_identifier_quotes_simple() -> None:
    """Verify simple identifiers are quoted."""
    expect_equal(quote_identifier("foo"), '"foo"')
    expect_equal(quote_identifier("Table_1"), '"Table_1"')


def test_quote_identifier_quotes_underscore_prefix() -> None:
    """Verify underscore-prefixed identifiers are quoted."""
    expect_equal(quote_identifier("_private"), '"_private"')


@pytest.mark.parametrize(
    "value",
    ["", "1foo", "foo-bar", "foo;drop", "foo bar", "foo.bar"],
)
def test_quote_identifier_rejects_unsafe(value: str) -> None:
    """Verify unsafe identifiers raise ValueError."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        quote_identifier(value)


# =============================================================================
# quote_table_key Tests
# =============================================================================


def test_quote_table_key_quotes_schema_and_table() -> None:
    """Verify schema.table format is properly quoted."""
    expect_equal(quote_table_key("core.modules"), '"core"."modules"')
    expect_equal(quote_table_key("analytics.metrics"), '"analytics"."metrics"')


@pytest.mark.parametrize(
    "value",
    ["", "table_only", "a..b", ".table", "schema.", "a.b.c"],
)
def test_quote_table_key_rejects_invalid(value: str) -> None:
    """Verify invalid table keys raise ValueError."""
    with pytest.raises(ValueError, match="Table key must include schema"):
        quote_table_key(value)


# =============================================================================
# macro_select_sql Tests
# =============================================================================


def test_macro_select_sql_builds_select() -> None:
    """Verify macro SELECT statement generation."""
    sql = macro_select_sql("metadata.dataset_rows", "?, ?")
    expect_equal(sql, 'SELECT * FROM /*metadata.dataset_rows*/ "metadata"."dataset_rows"(?, ?)')


def test_macro_select_sql_preserves_placeholders() -> None:
    """Verify placeholders are preserved in output."""
    sql = macro_select_sql("schema.macro", "?, ?, ?")
    expect_in("(?, ?, ?)", sql)


def test_macro_select_sql_rejects_unqualified() -> None:
    """Verify unqualified macro names raise ValueError."""
    with pytest.raises(ValueError, match="must include schema"):
        macro_select_sql("unqualified", "?")


# =============================================================================
# safe_macro_call Tests
# =============================================================================


def test_safe_macro_call_generates_sql_and_preserves_args() -> None:
    """Verify SQL generation and argument preservation."""
    sql, args = safe_macro_call("metadata.dataset_rows", [1, "test"])
    expect_query_contains(sql, "metadata.dataset_rows")
    expect_params(args, [1, "test"])


def test_safe_macro_call_validates_against_allowlist() -> None:
    """Verify allowlist validation."""
    allowed = {"metadata.dataset_rows", "metadata.other"}
    sql, _ = safe_macro_call("metadata.dataset_rows", [], allowed=allowed)
    expect_query_contains(sql, "dataset_rows")


def test_safe_macro_call_rejects_non_allowlisted() -> None:
    """Verify non-allowlisted macro raises ValueError."""
    allowed = {"metadata.dataset_rows"}
    with pytest.raises(ValueError, match="not allowlisted"):
        safe_macro_call("metadata.bad_macro", [], allowed=allowed)


def test_safe_macro_call_allows_any_without_allowlist() -> None:
    """Verify any macro is accepted when allowlist is None."""
    sql, _ = safe_macro_call("any.macro", [])
    expect_query_contains(sql, "any.macro")


# =============================================================================
# build_insert_sql Tests
# =============================================================================


def test_build_insert_sql_builds_basic() -> None:
    """Verify basic INSERT statement generation."""
    sql = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    expect_query_contains(
        sql, 'INSERT INTO "core"."modules"', '"module"', '"path"', "VALUES (?, ?, ?, ?)"
    )


def test_build_insert_sql_single_column() -> None:
    """Verify INSERT with single column."""
    sql = build_insert_sql("test.table", ["col"])
    expect_query_contains(sql, "VALUES (?)")


def test_build_insert_sql_identifier_already_quoted() -> None:
    """Verify pre-quoted identifier is preserved."""
    sql = build_insert_sql('"temp"."view"', ["col"], identifier_is_quoted=True)
    expect_query_contains(sql, '"temp"."view"')


def test_build_insert_sql_executes_on_real_db(fresh_gateway: StorageGateway) -> None:
    """Verify generated INSERT executes on real DuckDB."""
    sql = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    # Execute with test data
    fresh_gateway.con.execute(sql, ["test_mod", "test.py", "test/repo", "abc123"])
    # Verify insertion
    result = fresh_gateway.con.execute(
        "SELECT module FROM core.modules WHERE repo = ?", ["test/repo"]
    ).fetchone()
    row = require_row(result, message="Expected module row after insert")
    expect_equal(row[0], "test_mod")


# =============================================================================
# Integration Tests with Real DuckDB
# =============================================================================


def test_count_after_insert(fresh_gateway: StorageGateway) -> None:
    """Verify COUNT query after INSERT."""
    # Insert test data
    insert = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    fresh_gateway.con.execute(insert, ["mod1", "mod1.py", "test/repo", "abc"])
    fresh_gateway.con.execute(insert, ["mod2", "mod2.py", "test/repo", "abc"])

    # Count with WHERE
    query, params = QueryBuilder.count("core.modules", where={"repo": "test/repo"})
    result = fresh_gateway.con.execute(query, params).fetchone()
    row = require_row(result, message="Expected count row after insert")
    expect_equal(row[0], 2)


def test_delete_removes_rows(fresh_gateway: StorageGateway) -> None:
    """Verify DELETE query removes rows."""
    # Insert test data
    insert = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    fresh_gateway.con.execute(insert, ["mod1", "mod1.py", "test/repo", "abc"])

    # Delete
    query, params = QueryBuilder.delete("core.modules", where={"repo": "test/repo"})
    fresh_gateway.con.execute(query, params)

    # Verify deletion
    count_query, count_params = QueryBuilder.count("core.modules", where={"repo": "test/repo"})
    result = fresh_gateway.con.execute(count_query, count_params).fetchone()
    row = require_row(result, message="Expected count row after delete")
    expect_equal(row[0], 0)


def test_delete_in_removes_specific_rows(fresh_gateway: StorageGateway) -> None:
    """Verify DELETE IN removes specific rows."""
    # Insert test data
    insert = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    fresh_gateway.con.execute(insert, ["mod1", "mod1.py", "repo1", "abc"])
    fresh_gateway.con.execute(insert, ["mod2", "mod2.py", "repo2", "abc"])
    fresh_gateway.con.execute(insert, ["mod3", "mod3.py", "repo3", "abc"])

    # Delete specific repos
    query, params = QueryBuilder.delete_in("core.modules", "repo", ["repo1", "repo3"])
    fresh_gateway.con.execute(query, params)

    # Verify only repo2 remains
    select = QueryBuilder.select_all("core.modules")
    result = fresh_gateway.con.execute(select).fetchall()
    expect_length(result, 1, label="remaining_rows")
    expect_equal(result[0][2], "repo2")


def test_select_all_returns_data(fresh_gateway: StorageGateway) -> None:
    """Verify SELECT * returns inserted data."""
    # Insert test data
    insert = build_insert_sql("core.modules", ["module", "path", "repo", "commit"])
    fresh_gateway.con.execute(insert, ["test_mod", "test.py", "test/repo", "abc123"])

    # Select all
    query = QueryBuilder.select_all("core.modules")
    result = fresh_gateway.con.execute(query).fetchall()
    expect_length(result, 1, label="row_count")
    expect_equal(result[0][0], "test_mod")
