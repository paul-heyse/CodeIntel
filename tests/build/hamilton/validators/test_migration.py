"""Tests for Hamilton migration utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from codeintel.build.hamilton.validators import (
    ColumnsExistValidator,
    MigrationReport,
    NoNullsInColumnsValidator,
    UniqueColumnsValidator,
)
from codeintel.build.hamilton.validators.migration import (
    _pandera_dtype_to_hamilton_type,
    generate_migration_code,
    validators_from_pandera_schema,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)


class TestMigrationReport:
    """Test suite for MigrationReport class."""

    @staticmethod
    def test_initial_state() -> None:
        """Test report initial state."""
        report = MigrationReport("test.table")
        expect_equal(report.table_key, "test.table")
        expect_equal(report.columns_migrated, 0)
        expect_equal(report.validators_created, 0)
        expect_equal(report.warnings, [])
        expect_equal(report.errors, [])
        expect_true(report.success)

    @staticmethod
    def test_add_warning() -> None:
        """Test adding warnings."""
        report = MigrationReport("test.table")
        report.add_warning("Test warning")
        expect_length(report.warnings, 1)
        expect_in("Test warning", report.warnings)
        expect_true(report.success)  # Warnings don't cause failure

    @staticmethod
    def test_add_error() -> None:
        """Test adding errors."""
        report = MigrationReport("test.table")
        report.add_error("Test error")
        expect_length(report.errors, 1)
        expect_in("Test error", report.errors)
        expect_true(report.success is False)  # Errors cause failure

    @staticmethod
    def test_summary_success() -> None:
        """Test summary for successful migration."""
        report = MigrationReport("test.table")
        report.columns_migrated = 5
        report.validators_created = 3
        summary = report.summary()
        expect_in("SUCCESS", summary)
        expect_in("test.table", summary)
        expect_in("5 columns", summary)
        expect_in("3 validators", summary)

    @staticmethod
    def test_summary_failed() -> None:
        """Test summary for failed migration."""
        report = MigrationReport("test.table")
        report.add_error("Something went wrong")
        summary = report.summary()
        expect_in("FAILED", summary)


class TestPanderaDtypeConversion:
    """Test suite for dtype conversion."""

    @pytest.mark.parametrize(
        ("pandera_dtype", "expected"),
        [
            pytest.param("int64", "int", id="int64"),
            pytest.param("Int64", "int", id="Int64_nullable"),
            pytest.param("float64", "float", id="float64"),
            pytest.param("bool", "bool", id="bool"),
            pytest.param("datetime64[ns]", "datetime", id="datetime"),
            pytest.param("string", "string", id="string"),
            pytest.param("object", "string", id="object"),
            pytest.param("category", "object", id="unknown"),
        ],
    )
    @staticmethod
    def test_dtype_conversion(pandera_dtype: str, expected: str) -> None:
        """Test Pandera dtype to Hamilton type conversion."""
        result = _pandera_dtype_to_hamilton_type(pandera_dtype)
        expect_equal(result, expected)


class TestValidatorsFromPanderaSchema:
    """Test suite for validators_from_pandera_schema."""

    @staticmethod
    def test_basic_schema() -> None:
        """Test converting a basic Pandera schema."""
        # Create mock Pandera schema
        mock_column = MagicMock()
        mock_column.dtype = "int64"
        mock_column.nullable = True
        mock_column.unique = False

        mock_schema = MagicMock()
        mock_schema.columns = {"id": mock_column}

        validators = validators_from_pandera_schema(mock_schema)

        # Should have ColumnsExistValidator and ColumnTypesValidator
        validator_types = [type(v).__name__ for v in validators]
        expect_in("ColumnsExistValidator", validator_types)
        expect_in("ColumnTypesValidator", validator_types)

    @staticmethod
    def test_schema_with_constraints() -> None:
        """Test converting schema with nullable and unique constraints."""
        mock_id_col = MagicMock()
        mock_id_col.dtype = "int64"
        mock_id_col.nullable = False  # Not nullable
        mock_id_col.unique = True  # Must be unique

        mock_name_col = MagicMock()
        mock_name_col.dtype = "string"
        mock_name_col.nullable = True
        mock_name_col.unique = False

        mock_schema = MagicMock()
        mock_schema.columns = {
            "id": mock_id_col,
            "name": mock_name_col,
        }

        validators = validators_from_pandera_schema(mock_schema)

        # Check we got the expected validators
        columns_exist = None
        no_nulls = None
        unique = None
        for v in validators:
            if isinstance(v, ColumnsExistValidator):
                columns_exist = v
            elif isinstance(v, NoNullsInColumnsValidator):
                no_nulls = v
            elif isinstance(v, UniqueColumnsValidator):
                unique = v

        expect_true(columns_exist is not None)
        expect_in("id", columns_exist.required_columns)
        expect_in("name", columns_exist.required_columns)

        expect_true(no_nulls is not None)
        expect_in("id", no_nulls.columns)
        expect_true("name" not in no_nulls.columns)

        expect_true(unique is not None)
        expect_in("id", unique.columns)

    @staticmethod
    def test_empty_schema() -> None:
        """Test converting empty schema."""
        mock_schema = MagicMock()
        mock_schema.columns = {}

        validators = validators_from_pandera_schema(mock_schema)
        expect_equal(validators, [])


class TestGenerateMigrationCode:
    """Test suite for generate_migration_code."""

    @staticmethod
    def test_generates_valid_code_structure() -> None:
        """Test that generated code has expected structure."""
        code = generate_migration_code("analytics.test_table")

        # Check imports are present
        expect_in("from hamilton.function_modifiers import", code)
        expect_in("check_output_custom", code)
        expect_in("schema", code)
        expect_in("tag", code)

        # Check decorator structure
        expect_in("@tag(", code)
        expect_in("@schema.output(", code)
        expect_in("@check_output_custom(", code)

        # Check domain and target are extracted
        expect_in('domain="analytics"', code)
        expect_in('target="test_table"', code)

    @staticmethod
    def test_custom_node_name() -> None:
        """Test generating code with custom node name."""
        code = generate_migration_code(
            "analytics.test_table",
            node_name="my_custom_compute",
        )
        expect_in("def my_custom_compute(", code)

    @staticmethod
    def test_default_node_name() -> None:
        """Test default node name generation."""
        code = generate_migration_code("analytics.test_table")
        expect_in("def t__analytics__test_table__compute(", code)


@pytest.mark.parametrize(
    ("columns", "expected_validator_count"),
    [
        pytest.param({}, 0, id="empty"),
        pytest.param(
            {"id": MagicMock(dtype="int64", nullable=True, unique=False)},
            2,  # ColumnsExist + ColumnTypes
            id="basic_column",
        ),
        pytest.param(
            {"id": MagicMock(dtype="int64", nullable=False, unique=True)},
            4,  # ColumnsExist + ColumnTypes + NoNulls + Unique
            id="constrained_column",
        ),
    ],
)
def test_validator_count_from_schema(
    columns: dict[str, Any],
    expected_validator_count: int,
) -> None:
    """Parametrized test for validator count based on schema."""
    mock_schema = MagicMock()
    mock_schema.columns = columns
    validators = validators_from_pandera_schema(mock_schema)
    expect_length(validators, expected_validator_count)
