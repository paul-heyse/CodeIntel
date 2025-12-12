"""Tests for codeintel.config.datasets.sql module."""

from __future__ import annotations

import pytest

from codeintel.config.datasets.sql import (
    AST_NODES_DELETE,
    CFG_BLOCKS_DELETE,
    FILE_STATE_DELETE,
    build_delete_sql,
    build_delete_sql_by_table,
    build_insert_sql,
    build_insert_sql_by_table,
    get_contract_columns,
    get_delete_sql_by_table,
    get_insert_sql_by_table,
    get_table_columns,
    load_columns_by_table,
    serialize_row,
)


def require(*, condition: object, message: str) -> None:
    """Fail the current test with a descriptive message."""
    if not condition:
        pytest.fail(message)


def test_build_insert_sql_valid_table() -> None:
    """Verify build_insert_sql generates correct INSERT statement."""
    sql = build_insert_sql("core.goids")
    require(
        condition=sql.startswith("INSERT INTO core.goids"),
        message="INSERT statement should target table",
    )
    require(condition="VALUES" in sql, message="INSERT statement should include VALUES clause")
    require(condition="?" in sql, message="INSERT statement should be parameterized")


def test_build_insert_sql_invalid_table_raises() -> None:
    """Verify build_insert_sql raises for unknown table."""
    with pytest.raises(ValueError, match="No schema defined"):
        build_insert_sql("nonexistent.table")


def test_build_delete_sql_with_repo_commit() -> None:
    """Verify build_delete_sql generates correct DELETE for repo/commit tables."""
    sql = build_delete_sql("analytics.function_metrics")

    if sql is None:
        pytest.fail("expected delete sql for analytics.function_metrics")
    require(condition="DELETE FROM" in sql, message="delete statement should include DELETE FROM")
    require(
        condition="WHERE repo = ? AND commit = ?" in sql,
        message="delete statement should filter repo/commit",
    )


def test_build_delete_sql_returns_none_for_no_repo_commit() -> None:
    """Verify build_delete_sql returns None for tables without repo/commit."""
    result = build_delete_sql("core.ast_nodes")
    require(
        condition=result is None,
        message="tables without repo/commit should not produce delete SQL",
    )


def test_build_insert_sql_by_table_returns_dict() -> None:
    """Verify build_insert_sql_by_table returns a dictionary."""
    sql_dict = build_insert_sql_by_table()
    require(
        condition=isinstance(sql_dict, dict),
        message="build_insert_sql_by_table should return a dict",
    )
    require(condition=len(sql_dict) > 0, message="insert SQL dictionary should not be empty")


def test_build_insert_sql_by_table_excludes_views() -> None:
    """Verify build_insert_sql_by_table excludes docs.* views."""
    sql_dict = build_insert_sql_by_table()
    for key in sql_dict:
        require(
            condition=not key.startswith("docs."),
            message=f"docs view {key} should not have INSERT SQL",
        )


def test_build_delete_sql_by_table_returns_dict() -> None:
    """Verify build_delete_sql_by_table returns a dictionary."""
    sql_dict = build_delete_sql_by_table()
    require(
        condition=isinstance(sql_dict, dict),
        message="build_delete_sql_by_table should return a dict",
    )

    insert_dict = build_insert_sql_by_table()
    require(
        condition=len(sql_dict) <= len(insert_dict),
        message="delete SQL dictionary should not exceed insert SQL dictionary",
    )


def test_get_insert_sql_by_table_lazy_loading() -> None:
    """Verify get_insert_sql_by_table returns same dict as build function."""
    lazy_dict = get_insert_sql_by_table()
    fresh_dict = build_insert_sql_by_table()
    require(
        condition=lazy_dict == fresh_dict,
        message="lazy insert SQL should equal freshly built dictionary",
    )


def test_get_delete_sql_by_table_lazy_loading() -> None:
    """Verify get_delete_sql_by_table returns same dict as build function."""
    lazy_dict = get_delete_sql_by_table()
    fresh_dict = build_delete_sql_by_table()
    require(
        condition=lazy_dict == fresh_dict,
        message="lazy delete SQL should equal freshly built dictionary",
    )


def test_special_delete_constants_exist() -> None:
    """Verify special DELETE constants are defined correctly."""
    require(
        condition="DELETE FROM core.ast_nodes" in AST_NODES_DELETE,
        message="AST_NODES_DELETE constant missing",
    )
    require(
        condition="DELETE FROM graph.cfg_blocks" in CFG_BLOCKS_DELETE,
        message="CFG_BLOCKS_DELETE missing",
    )
    require(
        condition="DELETE FROM core.file_state" in FILE_STATE_DELETE,
        message="FILE_STATE_DELETE missing",
    )


def test_load_columns_by_table_returns_dict() -> None:
    """Verify load_columns_by_table returns column lists for all tables."""
    columns = load_columns_by_table()
    require(
        condition=isinstance(columns, dict),
        message="load_columns_by_table should return a dict",
    )
    require(condition=len(columns) > 0, message="columns mapping should not be empty")

    for table_key, col_list in columns.items():
        require(condition=isinstance(col_list, list), message=f"{table_key} columns is not a list")
        for col_name in col_list:
            require(
                condition=isinstance(col_name, str),
                message=f"{table_key}.{col_name} is not a string",
            )


def test_get_table_columns_valid_table() -> None:
    """Verify get_table_columns returns column names for valid table."""
    columns = get_table_columns("core.goids")
    require(condition=isinstance(columns, list), message="table columns should be a list")
    require(condition=len(columns) > 0, message="table columns should not be empty")
    require(condition="goid_h128" in columns, message="goid_h128 column should be present")
    require(condition="urn" in columns, message="urn column should be present")


def test_get_table_columns_invalid_table_raises() -> None:
    """Verify get_table_columns raises for unknown table."""
    with pytest.raises(KeyError, match="No schema defined"):
        get_table_columns("nonexistent.table")


def test_get_contract_columns_valid_table() -> None:
    """Verify get_contract_columns returns tuple for valid table."""
    columns = get_contract_columns("core.goids")
    require(condition=isinstance(columns, tuple), message="contract columns should be a tuple")
    require(condition=len(columns) > 0, message="contract columns should not be empty")


def test_get_contract_columns_invalid_table_raises() -> None:
    """Verify get_contract_columns raises for unknown table."""
    with pytest.raises(ValueError, match="No schema defined"):
        get_contract_columns("nonexistent.table")


def test_serialize_row_basic() -> None:
    """Verify serialize_row converts dict to tuple in column order."""
    row = {"a": 1, "b": 2, "c": 3}
    columns = ["c", "a", "b"]
    result = serialize_row(row, columns)
    require(
        condition=result == (3, 1, 2),
        message="serialize_row should follow provided column order",
    )


def test_serialize_row_preserves_values() -> None:
    """Verify serialize_row preserves value types."""
    row = {"int_col": 42, "str_col": "hello", "none_col": None, "bool_col": True}
    columns = ["int_col", "str_col", "none_col", "bool_col"]
    result = serialize_row(row, columns)
    require(
        condition=result == (42, "hello", None, True),
        message="serialize_row should preserve value types and order",
    )
