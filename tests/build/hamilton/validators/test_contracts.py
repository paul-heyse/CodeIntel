"""Tests for Hamilton-native contract builders.

These tests verify that the contract builder functions correctly
create validator sets for common patterns.
"""
from __future__ import annotations

import pandas as pd
import pytest

from codeintel.build.hamilton.validators import (
    ColumnsExistValidator,
    build_enum_column_contract,
    build_key_column_contract,
    build_metrics_contract,
    build_table_contract,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_not_in,
    expect_true,
)


class TestBuildTableContract:
    """Test suite for build_table_contract function."""

    @staticmethod
    def test_minimal_contract() -> None:
        """Test contract with only required columns."""
        validators = build_table_contract(
            required_columns=["id", "name"],
        )
        expect_equal(len(validators), 1)
        expect_is_instance(validators[0], ColumnsExistValidator)

    @staticmethod
    def test_full_contract() -> None:
        """Test contract with all options."""
        validators = build_table_contract(
            required_columns=["id", "name", "value"],
            column_types={"id": "int", "value": "float"},
            no_nulls=["id", "name"],
            unique=["id"],
            min_rows=1,
            max_rows=1000,
        )

        # Should have 5 validators
        expect_equal(len(validators), 5)

        # Check types
        validator_types = [type(v).__name__ for v in validators]
        expect_in("ColumnsExistValidator", validator_types)
        expect_in("ColumnTypesValidator", validator_types)
        expect_in("NoNullsInColumnsValidator", validator_types)
        expect_in("UniqueColumnsValidator", validator_types)
        expect_in("RowCountRangeValidator", validator_types)

    @staticmethod
    def test_contract_validates_dataframe() -> None:
        """Test that contract correctly validates a DataFrame."""
        validators = build_table_contract(
            required_columns=["id", "name"],
            no_nulls=["id"],
            unique=["id"],
        )

        # Valid DataFrame
        df_valid = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        })
        results = [v.validate(df_valid) for v in validators]
        expect_true(all(r.passes for r in results))

        # Invalid DataFrame (duplicate id)
        df_invalid = pd.DataFrame({
            "id": [1, 1, 3],
            "name": ["a", "b", "c"],
        })
        results = [v.validate(df_invalid) for v in validators]
        expect_false(all(r.passes for r in results))

    @staticmethod
    def test_no_row_count_validator_when_defaults() -> None:
        """Test that no row count validator is added with default min=0."""
        validators = build_table_contract(
            required_columns=["id"],
            min_rows=0,
            max_rows=None,
        )
        validator_types = [type(v).__name__ for v in validators]
        expect_not_in("RowCountRangeValidator", validator_types)


class TestBuildKeyColumnContract:
    """Test suite for build_key_column_contract function."""

    @staticmethod
    def test_single_key_column() -> None:
        """Test contract with single key column."""
        validators = build_key_column_contract(
            key_columns=["id"],
        )

        # Should have 3 validators: exists, no_nulls, unique
        expect_equal(len(validators), 3)

    @staticmethod
    def test_composite_key() -> None:
        """Test contract with composite key columns."""
        validators = build_key_column_contract(
            key_columns=["repo", "commit", "path"],
        )

        # Should have 2 validators: exists, no_nulls (no unique for composite)
        expect_equal(len(validators), 2)

    @staticmethod
    def test_with_additional_columns() -> None:
        """Test contract with additional non-key columns."""
        validators = build_key_column_contract(
            key_columns=["id"],
            additional_columns=["name", "value"],
        )

        # Exists validator should include all columns
        expect_is_instance(validators[0], ColumnsExistValidator)

    @staticmethod
    def test_key_contract_validates_dataframe() -> None:
        """Test that key contract validates correctly."""
        validators = build_key_column_contract(
            key_columns=["id"],
            additional_columns=["name"],
        )

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        })

        results = [v.validate(df) for v in validators]
        expect_true(all(r.passes for r in results))


class TestBuildMetricsContract:
    """Test suite for build_metrics_contract function."""

    @staticmethod
    def test_basic_metrics_contract() -> None:
        """Test basic metrics contract."""
        validators = build_metrics_contract(
            metric_columns=["loc", "complexity", "coverage"],
        )

        expect_equal(len(validators), 1)
        expect_is_instance(validators[0], ColumnsExistValidator)

    @staticmethod
    def test_metrics_with_types() -> None:
        """Test metrics contract with type specifications."""
        validators = build_metrics_contract(
            metric_columns=["loc", "complexity"],
            metric_types={"loc": "int", "complexity": "int"},
        )

        expect_equal(len(validators), 2)
        validator_types = [type(v).__name__ for v in validators]
        expect_in("ColumnTypesValidator", validator_types)

    @staticmethod
    def test_metrics_contract_validates_dataframe() -> None:
        """Test that metrics contract validates correctly."""
        validators = build_metrics_contract(
            metric_columns=["loc", "complexity"],
            metric_types={"loc": "int"},
        )

        df = pd.DataFrame({
            "loc": [10, 20, 30],
            "complexity": [1, 2, 3],
        })

        results = [v.validate(df) for v in validators]
        expect_true(all(r.passes for r in results))


class TestBuildEnumColumnContract:
    """Test suite for build_enum_column_contract function."""

    @staticmethod
    def test_basic_enum_contract() -> None:
        """Test basic enum column contract."""
        validators = build_enum_column_contract(
            column="status",
            allowed_values={"active", "inactive"},
        )

        # Should have 3 validators: exists, values_in_set, no_nulls
        expect_equal(len(validators), 3)

    @staticmethod
    def test_enum_allows_nulls() -> None:
        """Test enum contract that allows nulls."""
        validators = build_enum_column_contract(
            column="status",
            allowed_values={"active", "inactive"},
            allow_nulls=True,
        )

        # Should have 2 validators: exists, values_in_set
        expect_equal(len(validators), 2)

    @staticmethod
    def test_enum_contract_validates_dataframe() -> None:
        """Test that enum contract validates correctly."""
        validators = build_enum_column_contract(
            column="status",
            allowed_values={"active", "inactive"},
        )

        # Valid
        df_valid = pd.DataFrame({"status": ["active", "inactive", "active"]})
        results = [v.validate(df_valid) for v in validators]
        expect_true(all(r.passes for r in results))

        # Invalid
        df_invalid = pd.DataFrame({"status": ["active", "pending"]})
        results = [v.validate(df_invalid) for v in validators]
        expect_false(all(r.passes for r in results))


@pytest.mark.parametrize(
    ("min_rows", "max_rows", "row_count", "expected_pass"),
    [
        pytest.param(0, None, 0, True, id="empty_no_constraints"),
        pytest.param(1, None, 5, True, id="min_1_has_5"),
        pytest.param(10, None, 5, False, id="min_10_has_5"),
        pytest.param(0, 10, 5, True, id="max_10_has_5"),
        pytest.param(0, 10, 15, False, id="max_10_has_15"),
        pytest.param(5, 10, 7, True, id="range_5_10_has_7"),
    ],
)
def test_table_contract_row_count_parametrized(
    min_rows: int,
    max_rows: int | None,
    row_count: int,
    expected_pass: bool,
) -> None:
    """Parametrized test for row count validation in table contracts."""
    validators = build_table_contract(
        required_columns=["id"],
        min_rows=min_rows,
        max_rows=max_rows,
    )

    df = pd.DataFrame({"id": range(row_count)})
    results = [v.validate(df) for v in validators]

    # The row count validator may not be present if min=0 and max=None
    if min_rows == 0 and max_rows is None:
        # No row count validator, should pass
        expect_true(all(r.passes for r in results))
    else:
        expect_equal(all(r.passes for r in results), expected_pass)
