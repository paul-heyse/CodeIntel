"""Tests for column lineage tracing.

Tests the ColumnLineage and TableLineage dataclasses,
trace_column_lineage and trace_table_lineage functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.hamilton.contracts.schemas.constraints import Constraint, ConstraintKind
from codeintel.build.hamilton.contracts.schemas.lineage import (
    ColumnLineage,
    TableLineage,
    get_all_columns_with_constraint,
    trace_column_lineage,
    trace_table_lineage,
)

if TYPE_CHECKING:
    from collections.abc import Container


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


def _expect_in(item: object, container: Container[object], label: str) -> None:
    """Check item is in container with clear failure message."""
    if item not in container:
        pytest.fail(f"{label}: {item!r} not in {container!r}")


def test_column_lineage_creation() -> None:
    """Create ColumnLineage with basic fields."""
    lineage = ColumnLineage(
        column="loc",
        table_key="analytics.function_metrics",
        constraints=[],
        producer_plugins=[],
        upstream_columns=[],
    )
    _expect_equal(lineage.column, "loc", "column")
    _expect_equal(lineage.table_key, "analytics.function_metrics", "table_key")
    _expect_equal(len(lineage.constraints), 0, "constraints")
    _expect_equal(len(lineage.producer_plugins), 0, "producer_plugins")
    _expect_equal(len(lineage.upstream_columns), 0, "upstream_columns")


def test_column_lineage_with_constraints() -> None:
    """Create ColumnLineage with constraints."""
    type_constraint = Constraint(
        kind=ConstraintKind.TYPE,
        column="loc",
        expression="loc: int",
    )
    range_constraint = Constraint(
        kind=ConstraintKind.RANGE,
        column="loc",
        expression="loc >= 0",
    )

    lineage = ColumnLineage(
        column="loc",
        table_key="analytics.function_metrics",
        constraints=[type_constraint, range_constraint],
        producer_plugins=["analytics.function_metrics"],
        upstream_columns=[],
    )

    _expect_equal(len(lineage.constraints), 2, "constraints count")
    _require(condition=lineage.has_type_constraint, message="should have type constraint")
    _require(condition=lineage.has_range_constraint, message="should have range constraint")


def test_column_lineage_is_nullable_true() -> None:
    """Verify is_nullable returns True for nullable columns."""
    constraint = Constraint(
        kind=ConstraintKind.NULLABILITY,
        column="optional_col",
        expression="optional_col nullable",
    )
    lineage = ColumnLineage(
        column="optional_col",
        table_key="test.table",
        constraints=[constraint],
        producer_plugins=[],
        upstream_columns=[],
    )
    _require(condition=lineage.is_nullable is True, message="should be nullable")


def test_column_lineage_is_nullable_false() -> None:
    """Verify is_nullable returns False for required columns."""
    constraint = Constraint(
        kind=ConstraintKind.NULLABILITY,
        column="required_col",
        expression="required_col required",
    )
    lineage = ColumnLineage(
        column="required_col",
        table_key="test.table",
        constraints=[constraint],
        producer_plugins=[],
        upstream_columns=[],
    )
    _require(condition=lineage.is_nullable is False, message="should not be nullable")


def test_column_lineage_is_nullable_none() -> None:
    """Verify is_nullable returns None when no nullability constraint."""
    lineage = ColumnLineage(
        column="col",
        table_key="test.table",
        constraints=[],
        producer_plugins=[],
        upstream_columns=[],
    )
    _require(condition=lineage.is_nullable is None, message="should be None")


def test_column_lineage_summary() -> None:
    """Verify summary() returns a readable string."""
    constraint = Constraint(
        kind=ConstraintKind.TYPE,
        column="loc",
        expression="loc: int",
    )
    lineage = ColumnLineage(
        column="loc",
        table_key="analytics.function_metrics",
        constraints=[constraint],
        producer_plugins=["test_plugin"],
        upstream_columns=[],
    )

    summary = lineage.summary()
    _require(
        condition="Column: analytics.function_metrics.loc" in summary, message="should have header"
    )
    _require(condition="loc: int" in summary, message="should have constraint")
    _require(condition="test_plugin" in summary, message="should have producer")


def test_table_lineage_creation() -> None:
    """Create TableLineage with basic fields."""
    lineage = TableLineage(
        table_key="analytics.function_metrics",
        columns={},
        producer_plugins=[],
        upstream_tables=[],
    )
    _expect_equal(lineage.table_key, "analytics.function_metrics", "table_key")
    _expect_equal(lineage.column_count, 0, "column_count")


def test_table_lineage_with_columns() -> None:
    """Create TableLineage with column lineages."""
    col_lineage = ColumnLineage(
        column="loc",
        table_key="test.table",
        constraints=[],
        producer_plugins=[],
        upstream_columns=[],
    )
    lineage = TableLineage(
        table_key="test.table",
        columns={"loc": col_lineage},
        producer_plugins=["test_plugin"],
        upstream_tables=["core.goids"],
    )

    _expect_equal(lineage.column_count, 1, "column_count")
    _expect_equal(len(lineage.producer_plugins), 1, "producer_plugins")
    _expect_equal(len(lineage.upstream_tables), 1, "upstream_tables")


def test_table_lineage_get_column() -> None:
    """Verify get_column returns correct lineage."""
    col_lineage = ColumnLineage(
        column="loc",
        table_key="test.table",
        constraints=[],
        producer_plugins=[],
        upstream_columns=[],
    )
    lineage = TableLineage(
        table_key="test.table",
        columns={"loc": col_lineage},
        producer_plugins=[],
        upstream_tables=[],
    )

    result = lineage.get_column("loc")
    if result is None:
        pytest.fail("should find column")
    _expect_equal(result.column, "loc", "column name")


def test_table_lineage_get_column_missing() -> None:
    """Verify get_column returns None for missing column."""
    lineage = TableLineage(
        table_key="test.table",
        columns={},
        producer_plugins=[],
        upstream_tables=[],
    )

    result = lineage.get_column("nonexistent")
    _require(condition=result is None, message="should return None")


def test_trace_column_lineage_missing_table() -> None:
    """Verify trace_column_lineage raises KeyError for missing table."""
    try:
        trace_column_lineage("nonexistent.table", "col")
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass


def test_trace_column_lineage_registered_table() -> None:
    """Verify trace_column_lineage works for registered tables."""
    if len(SCHEMA_REGISTRY) == 0:
        pytest.skip("No schemas registered")

    table_key = next(iter(SCHEMA_REGISTRY.keys()))
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is None:
        pytest.skip("Could not get schema")

    columns = schema.column_names()
    if not columns:
        pytest.skip("No columns in schema")

    first_col = columns[0]
    lineage = trace_column_lineage(table_key, first_col)

    _expect_equal(lineage.table_key, table_key, "table_key")
    _expect_equal(lineage.column, first_col, "column")
    _require(condition=isinstance(lineage.constraints, list), message="constraints should be list")


def test_trace_column_lineage_missing_column() -> None:
    """Verify trace_column_lineage raises ValueError for missing column."""
    if len(SCHEMA_REGISTRY) == 0:
        pytest.skip("No schemas registered")

    table_key = next(iter(SCHEMA_REGISTRY.keys()))

    try:
        trace_column_lineage(table_key, "nonexistent_column_xyz")
        pytest.fail("Should have raised ValueError")
    except ValueError:
        pass


def test_trace_table_lineage_missing_table() -> None:
    """Verify trace_table_lineage raises KeyError for missing table."""
    try:
        trace_table_lineage("nonexistent.table")
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass


def test_trace_table_lineage_registered_table() -> None:
    """Verify trace_table_lineage works for registered tables."""
    if len(SCHEMA_REGISTRY) == 0:
        pytest.skip("No schemas registered")

    table_key = next(iter(SCHEMA_REGISTRY.keys()))
    lineage = trace_table_lineage(table_key)

    _expect_equal(lineage.table_key, table_key, "table_key")
    _require(condition=isinstance(lineage.columns, dict), message="columns should be dict")
    _require(condition=lineage.column_count >= 0, message="column_count should be non-negative")


def test_get_all_columns_with_constraint_returns_list() -> None:
    """Verify get_all_columns_with_constraint returns a list."""
    result = get_all_columns_with_constraint("type")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_all_columns_with_constraint_invalid_kind() -> None:
    """Verify get_all_columns_with_constraint returns empty for invalid kind."""
    result = get_all_columns_with_constraint("invalid_kind")
    _expect_equal(len(result), 0, "should be empty for invalid kind")


def test_get_all_columns_with_constraint_tuple_format() -> None:
    """Verify results are (table_key, column) tuples."""
    expected_tuple_length = 2
    result = get_all_columns_with_constraint("type")
    for item in result:
        _require(condition=isinstance(item, tuple), message="should be tuple")
        _require(condition=len(item) == expected_tuple_length, message="should have 2 elements")
        table_key, col = item
        _require(condition=isinstance(table_key, str), message="table_key should be str")
        _require(condition=isinstance(col, str), message="column should be str")
