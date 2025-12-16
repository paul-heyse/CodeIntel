"""Tests for Hamilton-native DataFrame validators.

These tests verify that the custom validators correctly validate
DataFrame outputs according to Hamilton's validation framework.
"""
from __future__ import annotations

import pandas as pd
import pytest

from codeintel.build.hamilton.validators import (
    ColumnsExistValidator,
    ColumnTypesValidator,
    ColumnValuesInSetValidator,
    NoNullsInColumnsValidator,
    RowCountRangeValidator,
    RowCountValidator,
    UniqueColumnsValidator,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_true,
)


class TestColumnsExistValidator:
    """Test suite for ColumnsExistValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(ColumnsExistValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_applies_to_other_types() -> None:
        """Validate applies_to returns True for all types (runtime checking).

        Note: applies_to now returns True for all types to support both
        pandas DataFrames and Ibis tables. Runtime type checking happens
        in the validate() method which returns a skipped result for
        unsupported types.
        """
        # Changed in Phase 1.5: applies_to returns True for all types
        # Runtime type checking handles unsupported types gracefully
        expect_true(ColumnsExistValidator.applies_to(list))
        expect_true(ColumnsExistValidator.applies_to(dict))

    @staticmethod
    def test_all_columns_exist() -> None:
        """Validate passes when all required columns exist."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [10, 20]})
        validator = ColumnsExistValidator(["id", "name"])
        result = validator.validate(df)
        expect_true(result.passes)
        expect_in("required columns present", result.message)

    @staticmethod
    def test_missing_columns() -> None:
        """Validate fails when columns are missing."""
        df = pd.DataFrame({"id": [1, 2]})
        validator = ColumnsExistValidator(["id", "name", "value"])
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("Missing required columns", result.message)
        expect_in("name", result.diagnostics["missing_columns"])
        expect_in("value", result.diagnostics["missing_columns"])

    @staticmethod
    def test_empty_columns_list() -> None:
        """Validate passes with empty columns list."""
        df = pd.DataFrame({"id": [1]})
        validator = ColumnsExistValidator([])
        result = validator.validate(df)
        expect_true(result.passes)


class TestColumnTypesValidator:
    """Test suite for ColumnTypesValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(ColumnTypesValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_correct_types() -> None:
        """Validate passes when column types match."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        validator = ColumnTypesValidator({"id": "int", "name": "string"})
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_type_mismatch() -> None:
        """Validate fails when column types don't match."""
        df = pd.DataFrame({"id": ["1", "2"], "name": ["a", "b"]})
        validator = ColumnTypesValidator({"id": "int"})
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("id", result.diagnostics["mismatches"])

    @staticmethod
    def test_missing_column_skipped() -> None:
        """Validate missing columns are skipped."""
        df = pd.DataFrame({"id": [1, 2]})
        validator = ColumnTypesValidator({"id": "int", "missing": "string"})
        result = validator.validate(df)
        expect_true(result.passes)


class TestNoNullsInColumnsValidator:
    """Test suite for NoNullsInColumnsValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(NoNullsInColumnsValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_no_nulls() -> None:
        """Validate passes when columns have no nulls."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        validator = NoNullsInColumnsValidator(["id", "name"])
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_has_nulls() -> None:
        """Validate fails when columns have nulls."""
        df = pd.DataFrame({"id": [1, None, 3], "name": ["a", "b", None]})
        validator = NoNullsInColumnsValidator(["id", "name"])
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("id", result.diagnostics["null_counts"])
        expect_in("name", result.diagnostics["null_counts"])

    @staticmethod
    def test_missing_column_skipped() -> None:
        """Validate missing columns are skipped."""
        df = pd.DataFrame({"id": [1, 2]})
        validator = NoNullsInColumnsValidator(["id", "missing"])
        result = validator.validate(df)
        expect_true(result.passes)


class TestUniqueColumnsValidator:
    """Test suite for UniqueColumnsValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(UniqueColumnsValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_unique_values() -> None:
        """Validate passes when values are unique."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        validator = UniqueColumnsValidator(["id"])
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_duplicate_values() -> None:
        """Validate fails when values are duplicated."""
        df = pd.DataFrame({"id": [1, 1, 2], "name": ["a", "b", "c"]})
        validator = UniqueColumnsValidator(["id"])
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("id", result.diagnostics["duplicate_counts"])

    @staticmethod
    def test_missing_column_skipped() -> None:
        """Validate missing columns are skipped."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        validator = UniqueColumnsValidator(["id", "missing"])
        result = validator.validate(df)
        expect_true(result.passes)


class TestRowCountValidator:
    """Test suite for RowCountValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(RowCountValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_meets_minimum() -> None:
        """Validate passes when row count meets minimum."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        validator = RowCountValidator(min_rows=3)
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_below_minimum() -> None:
        """Validate fails when row count is below minimum."""
        df = pd.DataFrame({"id": [1, 2]})
        validator = RowCountValidator(min_rows=5)
        result = validator.validate(df)
        expect_false(result.passes)
        expect_equal(result.diagnostics["actual_rows"], 2)
        expect_equal(result.diagnostics["min_rows"], 5)


class TestRowCountRangeValidator:
    """Test suite for RowCountRangeValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(RowCountRangeValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_within_range() -> None:
        """Validate passes when row count is within range."""
        df = pd.DataFrame({"id": range(50)})
        validator = RowCountRangeValidator(min_rows=10, max_rows=100)
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_below_minimum() -> None:
        """Validate fails when row count is below minimum."""
        df = pd.DataFrame({"id": [1, 2]})
        validator = RowCountRangeValidator(min_rows=10, max_rows=100)
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("below minimum", result.message)

    @staticmethod
    def test_above_maximum() -> None:
        """Validate fails when row count exceeds maximum."""
        df = pd.DataFrame({"id": range(200)})
        validator = RowCountRangeValidator(min_rows=0, max_rows=100)
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("exceeds maximum", result.message)

    @staticmethod
    def test_no_maximum() -> None:
        """Validate passes with no maximum when above minimum."""
        df = pd.DataFrame({"id": range(10000)})
        validator = RowCountRangeValidator(min_rows=1, max_rows=None)
        result = validator.validate(df)
        expect_true(result.passes)


class TestColumnValuesInSetValidator:
    """Test suite for ColumnValuesInSetValidator."""

    @staticmethod
    def test_applies_to_dataframe() -> None:
        """Validate applies_to returns True for DataFrame."""
        expect_true(ColumnValuesInSetValidator.applies_to(pd.DataFrame))

    @staticmethod
    def test_all_values_valid() -> None:
        """Validate passes when all values are in allowed set."""
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        validator = ColumnValuesInSetValidator("status", {"active", "inactive"})
        result = validator.validate(df)
        expect_true(result.passes)

    @staticmethod
    def test_invalid_values() -> None:
        """Validate fails when values are not in allowed set."""
        df = pd.DataFrame({"status": ["active", "pending", "unknown"]})
        validator = ColumnValuesInSetValidator("status", {"active", "inactive"})
        result = validator.validate(df)
        expect_false(result.passes)
        expect_in("pending", result.diagnostics["invalid_values"])
        expect_in("unknown", result.diagnostics["invalid_values"])

    @staticmethod
    def test_missing_column_skipped() -> None:
        """Validate missing columns are skipped."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        validator = ColumnValuesInSetValidator("status", {"active"})
        result = validator.validate(df)
        expect_true(result.passes)
        expect_true(result.diagnostics.get("skipped"))

    @staticmethod
    def test_nulls_ignored() -> None:
        """Validate null values are ignored in set check."""
        df = pd.DataFrame({"status": ["active", None, "inactive"]})
        validator = ColumnValuesInSetValidator("status", {"active", "inactive"})
        result = validator.validate(df)
        expect_true(result.passes)


class TestValidatorIntegration:
    """Integration tests for combining validators."""

    @staticmethod
    def test_multiple_validators() -> None:
        """Test that multiple validators can be used together."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "status": ["active", "active", "inactive"],
        })

        validators = [
            ColumnsExistValidator(["id", "name", "status"]),
            NoNullsInColumnsValidator(["id"]),
            UniqueColumnsValidator(["id"]),
            ColumnValuesInSetValidator("status", {"active", "inactive"}),
        ]

        results = [v.validate(df) for v in validators]
        expect_true(all(r.passes for r in results))

    @staticmethod
    def test_failing_chain_stops_early_semantics() -> None:
        """Test behavior when early validator fails."""
        df = pd.DataFrame({"id": [1, 1, 3]})  # Duplicate id

        validators = [
            ColumnsExistValidator(["id"]),  # Pass
            UniqueColumnsValidator(["id"]),  # Fail
            RowCountValidator(min_rows=5),  # Would fail
        ]

        results = [v.validate(df) for v in validators]
        expect_true(results[0].passes)
        expect_false(results[1].passes)
        expect_false(results[2].passes)


@pytest.mark.parametrize(
    ("column_types", "expected_pass"),
    [
        pytest.param({"id": "int"}, True, id="int_column"),
        pytest.param({"name": "string"}, True, id="string_column"),
        pytest.param({"value": "float"}, True, id="float_column"),
        pytest.param({"active": "bool"}, True, id="bool_column"),
        pytest.param({"id": "string"}, False, id="wrong_type"),
    ],
)
def test_column_types_parametrized(
    column_types: dict[str, str],
    expected_pass: bool,
) -> None:
    """Parametrized test for column type validation."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [1.0, 2.0, 3.0],
        "active": [True, False, True],
    })
    validator = ColumnTypesValidator(column_types)
    result = validator.validate(df)
    expect_equal(result.passes, expected_pass)
