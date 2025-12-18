"""Rows pipeline helper function tests.

This module validates that the row conversion utilities in
``codeintel.build.hamilton.templates.materialize_template`` correctly convert
mapping rows to tuples in the specified column order.
"""

from __future__ import annotations

from typing import TypedDict

from codeintel.build.hamilton.templates.rows_helpers import row_to_tuple, rows_to_tuples
from tests._helpers.assertions.expectation_assertions import expect_equal


class SampleRow(TypedDict):
    """Sample TypedDict for testing row conversion."""

    id: int
    name: str
    value: float


# ============================================================================
# row_to_tuple Tests
# ============================================================================


def test_row_to_tuple_basic() -> None:
    """Verify row_to_tuple converts dict to tuple in column order."""
    columns = ("id", "name", "value")
    row: dict[str, object] = {"id": 1, "name": "test", "value": 42.5}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(1, "test", 42.5), label="row_to_tuple result")


def test_row_to_tuple_reordered_columns() -> None:
    """Verify row_to_tuple respects column order."""
    columns = ("value", "id", "name")
    row: dict[str, object] = {"id": 1, "name": "test", "value": 42.5}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(42.5, 1, "test"), label="reordered columns")


def test_row_to_tuple_missing_column() -> None:
    """Verify row_to_tuple returns None for missing columns."""
    columns = ("id", "name", "missing_col")
    row: dict[str, object] = {"id": 1, "name": "test"}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(1, "test", None), label="missing column")


def test_row_to_tuple_subset_columns() -> None:
    """Verify row_to_tuple selects only specified columns."""
    columns = ("id",)
    row: dict[str, object] = {"id": 1, "name": "test", "value": 42.5}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(1,), label="subset columns")


def test_row_to_tuple_empty_columns() -> None:
    """Verify row_to_tuple handles empty columns tuple."""
    columns: tuple[str, ...] = ()
    row: dict[str, object] = {"id": 1, "name": "test"}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(), label="empty columns")


def test_row_to_tuple_typed_dict() -> None:
    """Verify row_to_tuple works with TypedDict rows."""
    columns = ("id", "name", "value")
    row: SampleRow = {"id": 1, "name": "typed", "value": 99.9}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(1, "typed", 99.9), label="typed dict")


def test_row_to_tuple_none_value() -> None:
    """Verify row_to_tuple preserves None values in row."""
    columns = ("id", "name")
    row: dict[str, object] = {"id": 1, "name": None}

    result = row_to_tuple(row, columns)

    expect_equal(result, expected=(1, None), label="none value in row")


# ============================================================================
# rows_to_tuples Tests
# ============================================================================


def test_rows_to_tuples_basic() -> None:
    """Verify rows_to_tuples converts sequence of dicts to tuple of tuples."""
    columns = ("id", "name")
    rows: list[dict[str, object]] = [
        {"id": 1, "name": "first"},
        {"id": 2, "name": "second"},
        {"id": 3, "name": "third"},
    ]

    result = rows_to_tuples(rows, columns)

    expect_equal(len(result), expected=3, label="result length")
    expect_equal(result[0], expected=(1, "first"), label="first row")
    expect_equal(result[1], expected=(2, "second"), label="second row")
    expect_equal(result[2], expected=(3, "third"), label="third row")


def test_rows_to_tuples_empty() -> None:
    """Verify rows_to_tuples handles empty input."""
    columns = ("id", "name")
    rows: list[dict[str, object]] = []

    result = rows_to_tuples(rows, columns)

    expect_equal(result, expected=(), label="empty rows")


def test_rows_to_tuples_single_row() -> None:
    """Verify rows_to_tuples handles single row."""
    columns = ("id", "name")
    rows: list[dict[str, object]] = [{"id": 42, "name": "solo"}]

    result = rows_to_tuples(rows, columns)

    expect_equal(len(result), expected=1, label="result length")
    expect_equal(result[0], expected=(42, "solo"), label="single row")


def test_rows_to_tuples_missing_columns() -> None:
    """Verify rows_to_tuples handles rows with missing columns."""
    columns = ("id", "name", "value")
    rows: list[dict[str, object]] = [
        {"id": 1, "name": "full", "value": 10},
        {"id": 2, "name": "partial"},  # missing value
        {"id": 3},  # missing name and value
    ]

    result = rows_to_tuples(rows, columns)

    expect_equal(len(result), expected=3, label="result length")
    expect_equal(result[0], expected=(1, "full", 10), label="full row")
    expect_equal(result[1], expected=(2, "partial", None), label="partial row")
    expect_equal(result[2], expected=(3, None, None), label="minimal row")


def test_rows_to_tuples_typed_dict_list() -> None:
    """Verify rows_to_tuples works with list of TypedDict rows."""
    columns = ("id", "name", "value")
    rows: list[SampleRow] = [
        {"id": 1, "name": "typed1", "value": 1.1},
        {"id": 2, "name": "typed2", "value": 2.2},
    ]

    result = rows_to_tuples(rows, columns)

    expect_equal(len(result), expected=2, label="result length")
    expect_equal(result[0], expected=(1, "typed1", 1.1), label="first typed row")
    expect_equal(result[1], expected=(2, "typed2", 2.2), label="second typed row")


def test_rows_to_tuples_complex_values() -> None:
    """Verify rows_to_tuples handles complex value types."""
    columns = ("id", "tags", "nested")
    rows: list[dict[str, object]] = [
        {"id": 1, "tags": ["a", "b"], "nested": {"key": "val"}},
        {"id": 2, "tags": None, "nested": {"key2": "val2"}},
    ]

    result = rows_to_tuples(rows, columns)

    expect_equal(len(result), expected=2, label="result length")
    expect_equal(result[0], expected=(1, ["a", "b"], {"key": "val"}), label="first row")
    expect_equal(result[1], expected=(2, None, {"key2": "val2"}), label="second row")
