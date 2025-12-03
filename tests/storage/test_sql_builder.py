"""Tests for sql_builder module."""

from __future__ import annotations

import pytest

from codeintel.storage.sql_builder import (
    InvalidIdentifierError,
    QueryBuilder,
    SafeColumn,
    SafeTable,
    SqlBuilderError,
    build_delete_query,
    render_sql,
    validate_identifier,
)


def test_validate_identifier_accepts_simple_name() -> None:
    """Verify validate_identifier accepts simple names."""
    result = validate_identifier("my_table")
    assert result == "my_table"


def test_validate_identifier_accepts_schema_qualified() -> None:
    """Verify validate_identifier accepts schema.table format."""
    result = validate_identifier("schema.table")
    assert result == "schema.table"


def test_validate_identifier_raises_on_empty() -> None:
    """Verify validate_identifier raises on empty string."""
    with pytest.raises(InvalidIdentifierError, match="cannot be empty"):
        validate_identifier("")


def test_validate_identifier_raises_on_too_long() -> None:
    """Verify validate_identifier raises when identifier exceeds max length."""
    max_identifier_length = 128
    long_name = "a" * (max_identifier_length + 1)
    with pytest.raises(InvalidIdentifierError, match="exceeds maximum length"):
        validate_identifier(long_name)


def test_validate_identifier_raises_on_invalid_chars() -> None:
    """Verify validate_identifier raises on invalid characters."""
    with pytest.raises(InvalidIdentifierError, match="must start with"):
        validate_identifier("1invalid")

    with pytest.raises(InvalidIdentifierError, match="must start with"):
        validate_identifier("table-name")

    with pytest.raises(InvalidIdentifierError, match="must start with"):
        validate_identifier("table name")


def test_safe_table_stores_valid_name() -> None:
    """Verify SafeTable stores valid table name."""
    table = SafeTable("analytics.metrics")
    assert str(table) == "analytics.metrics"


def test_safe_table_raises_on_invalid_name() -> None:
    """Verify SafeTable raises on invalid name."""
    with pytest.raises(InvalidIdentifierError):
        SafeTable("1invalid")


def test_safe_table_is_frozen() -> None:
    """Verify SafeTable is immutable."""
    table = SafeTable("core.modules")
    assert table.name == "core.modules"


def test_safe_column_stores_valid_name() -> None:
    """Verify SafeColumn stores valid column name."""
    col = SafeColumn("repo")
    assert str(col) == "repo"


def test_safe_column_raises_on_dots() -> None:
    """Verify SafeColumn raises on names with dots."""
    with pytest.raises(InvalidIdentifierError, match="cannot contain dots"):
        SafeColumn("schema.column")


def test_safe_column_raises_on_invalid_name() -> None:
    """Verify SafeColumn raises on invalid name."""
    with pytest.raises(InvalidIdentifierError):
        SafeColumn("1invalid")


def test_render_sql_joins_parts() -> None:
    """Verify render_sql joins parts with spaces."""
    result = render_sql(["SELECT", "*", "FROM", "table"])
    assert result == "SELECT * FROM table"


def test_render_sql_filters_empty_strings() -> None:
    """Verify render_sql filters out empty strings."""
    result = render_sql(["SELECT", "", "*", "", "FROM", "table"])
    assert result == "SELECT * FROM table"


def test_count_builds_basic_query() -> None:
    """Verify count builds basic COUNT query."""
    query, params = QueryBuilder.count("core.modules")
    assert "SELECT COUNT(*) FROM core.modules" in query
    assert not params


def test_count_with_where_clause() -> None:
    """Verify count builds query with WHERE clause."""
    query, params = QueryBuilder.count(
        "core.modules", where={"repo": "test/repo", "commit": "abc123"}
    )
    assert "WHERE" in query
    assert "repo = ?" in query
    assert "commit = ?" in query
    expected_param_count = 2
    assert len(params) == expected_param_count


def test_count_accepts_safe_table() -> None:
    """Verify count accepts SafeTable."""
    table = SafeTable("core.modules")
    query, _ = QueryBuilder.count(table)
    assert "core.modules" in query


def test_count_where_null_builds_correct_query() -> None:
    """Verify count_where_null builds IS NULL condition."""
    query, _ = QueryBuilder.count_where_null("core.modules", "language")
    assert "IS NULL" in query
    assert "language IS NULL" in query


def test_count_where_null_with_additional_where() -> None:
    """Verify count_where_null includes additional WHERE conditions."""
    query, params = QueryBuilder.count_where_null(
        "core.modules", "language", where={"repo": "test/repo"}
    )
    assert "language IS NULL" in query
    assert "repo = ?" in query
    assert len(params) == 1


def test_delete_builds_parameterized_query() -> None:
    """Verify delete builds DELETE with WHERE clause."""
    query, params = QueryBuilder.delete(
        "core.modules", where={"repo": "test/repo", "commit": "abc123"}
    )
    assert "DELETE FROM core.modules" in query
    assert "WHERE" in query
    expected_param_count = 2
    assert len(params) == expected_param_count


def test_delete_in_builds_in_clause() -> None:
    """Verify delete_in builds DELETE with IN clause."""
    query, params = QueryBuilder.delete_in("core.modules", "module", ["mod1", "mod2", "mod3"])
    assert "DELETE FROM core.modules" in query
    assert "module IN" in query
    assert "?, ?, ?" in query
    expected_param_count = 3
    assert len(params) == expected_param_count


def test_select_all_builds_query() -> None:
    """Verify select_all builds SELECT * query."""
    query = QueryBuilder.select_all("core.modules")
    assert query == "SELECT * FROM core.modules"


def test_select_all_validates_table() -> None:
    """Verify select_all validates table name."""
    with pytest.raises(InvalidIdentifierError):
        QueryBuilder.select_all("1invalid")


def test_insert_builds_placeholder_query() -> None:
    """Verify insert builds INSERT with placeholders."""
    query = QueryBuilder.insert("core.modules", ["module", "path", "repo", "commit"])
    assert "INSERT INTO core.modules" in query
    assert "(module, path, repo, commit)" in query
    assert "VALUES (?, ?, ?, ?)" in query


def test_insert_accepts_safe_columns() -> None:
    """Verify insert accepts SafeColumn objects."""
    columns = [SafeColumn("module"), SafeColumn("path")]
    query = QueryBuilder.insert("core.modules", columns)
    assert "(module, path)" in query


def test_delete_repo_commit_builds_scoped_query() -> None:
    """Verify delete_repo_commit builds repo/commit scoped delete."""
    query = QueryBuilder.delete_repo_commit("core.modules")
    assert "DELETE FROM core.modules" in query
    assert "repo = ?" in query
    assert "commit = ?" in query


def test_build_delete_query_with_scope() -> None:
    """Verify build_delete_query returns query when has_scope=True."""
    result = build_delete_query("core.modules", has_scope=True)
    assert result is not None
    assert "DELETE FROM core.modules" in result


def test_build_delete_query_without_scope() -> None:
    """Verify build_delete_query returns None when has_scope=False."""
    result = build_delete_query("core.modules", has_scope=False)
    assert result is None


def test_error_stores_identifier_and_reason() -> None:
    """Verify InvalidIdentifierError stores identifier and reason."""
    error = InvalidIdentifierError("bad_id", "test reason")
    assert error.identifier == "bad_id"
    assert error.reason == "test reason"


def test_error_message_includes_details() -> None:
    """Verify error message includes identifier and reason."""
    error = InvalidIdentifierError("bad_id", "test reason")
    message = str(error)
    assert "bad_id" in message
    assert "test reason" in message


def test_sql_builder_error_is_base_exception() -> None:
    """Verify SqlBuilderError is base for other exceptions."""
    assert issubclass(InvalidIdentifierError, SqlBuilderError)
