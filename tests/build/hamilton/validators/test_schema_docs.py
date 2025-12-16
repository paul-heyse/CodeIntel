"""Tests for Hamilton schema documentation utilities."""
from __future__ import annotations

import pytest

from codeintel.build.hamilton.schema_docs import (
    ColumnSchema,
    ColumnTypes,
    TableSchema,
    schema_from_columns,
    schema_output_tuple,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_length,
    expect_true,
)


class TestColumnSchema:
    """Test suite for ColumnSchema class."""

    @staticmethod
    def test_basic_creation() -> None:
        """Test creating a basic column schema."""
        col = ColumnSchema("id", "int")
        expect_equal(col.name, "id")
        expect_equal(col.dtype, "int")
        expect_true(col.description is None)
        expect_true(col.nullable)
        expect_true(col.unique is False)

    @staticmethod
    def test_full_creation() -> None:
        """Test creating a column schema with all options."""
        col = ColumnSchema(
            name="id",
            dtype="int",
            description="Primary key",
            nullable=False,
            unique=True,
        )
        expect_equal(col.name, "id")
        expect_equal(col.description, "Primary key")
        expect_true(col.nullable is False)
        expect_true(col.unique)

    @staticmethod
    def test_to_tuple() -> None:
        """Test conversion to schema.output tuple."""
        col = ColumnSchema("name", "string", description="User name")
        result = col.to_tuple()
        expect_equal(result, ("name", "string"))

    @staticmethod
    def test_to_dict_minimal() -> None:
        """Test conversion to dict with minimal fields."""
        col = ColumnSchema("value", "float")
        result = col.to_dict()
        expect_equal(result, {"name": "value", "dtype": "float"})

    @staticmethod
    def test_to_dict_full() -> None:
        """Test conversion to dict with all fields."""
        col = ColumnSchema(
            "id", "int",
            description="Primary key",
            nullable=False,
            unique=True,
        )
        result = col.to_dict()
        expect_equal(result, {
            "name": "id",
            "dtype": "int",
            "description": "Primary key",
            "nullable": False,
            "unique": True,
        })


class TestTableSchema:
    """Test suite for TableSchema class."""

    @staticmethod
    def test_basic_creation() -> None:
        """Test creating a basic table schema."""
        columns = [
            ColumnSchema("id", "int"),
            ColumnSchema("name", "string"),
        ]
        table = TableSchema("my.table", columns)
        expect_equal(table.table_key, "my.table")
        expect_length(table.columns, 2)
        expect_true(table.description is None)
        expect_equal(table.primary_key, [])

    @staticmethod
    def test_full_creation() -> None:
        """Test creating a table schema with all options."""
        columns = [
            ColumnSchema("id", "int", unique=True),
            ColumnSchema("name", "string"),
        ]
        table = TableSchema(
            "my.table",
            columns,
            description="Test table",
            primary_key=["id"],
        )
        expect_equal(table.description, "Test table")
        expect_equal(table.primary_key, ["id"])

    @staticmethod
    def test_schema_output_args() -> None:
        """Test generating @schema.output arguments."""
        columns = [
            ColumnSchema("id", "int"),
            ColumnSchema("name", "string"),
            ColumnSchema("value", "float"),
        ]
        table = TableSchema("test.table", columns)
        result = table.schema_output_args()
        expect_equal(result, (("id", "int"), ("name", "string"), ("value", "float")))

    @staticmethod
    def test_column_names() -> None:
        """Test getting column names."""
        columns = [
            ColumnSchema("a", "int"),
            ColumnSchema("b", "string"),
            ColumnSchema("c", "float"),
        ]
        table = TableSchema("test.table", columns)
        expect_equal(table.column_names(), ["a", "b", "c"])

    @staticmethod
    def test_non_nullable_columns() -> None:
        """Test getting non-nullable column names."""
        columns = [
            ColumnSchema("id", "int", nullable=False),
            ColumnSchema("name", "string", nullable=True),
            ColumnSchema("required", "string", nullable=False),
        ]
        table = TableSchema("test.table", columns)
        expect_equal(table.non_nullable_columns(), ["id", "required"])

    @staticmethod
    def test_unique_columns() -> None:
        """Test getting unique column names."""
        columns = [
            ColumnSchema("id", "int", unique=True),
            ColumnSchema("name", "string", unique=False),
            ColumnSchema("code", "string", unique=True),
        ]
        table = TableSchema("test.table", columns)
        expect_equal(table.unique_columns(), ["id", "code"])

    @staticmethod
    def test_to_dict() -> None:
        """Test converting table schema to dict."""
        columns = [ColumnSchema("id", "int")]
        table = TableSchema(
            "test.table",
            columns,
            description="Test",
            primary_key=["id"],
        )
        result = table.to_dict()
        expect_equal(result["table_key"], "test.table")
        expect_equal(result["description"], "Test")
        expect_equal(result["primary_key"], ["id"])
        expect_length(result["columns"], 1)


class TestSchemaHelpers:
    """Test suite for schema helper functions."""

    @staticmethod
    def test_schema_from_columns() -> None:
        """Test creating schema from column list."""
        columns = [
            ("id", "int"),
            ("name", "string"),
        ]
        result = schema_from_columns(columns)
        expect_equal(result, (("id", "int"), ("name", "string")))

    @staticmethod
    def test_schema_output_tuple() -> None:
        """Test creating schema from variadic args."""
        result = schema_output_tuple(
            ("id", "int"),
            ("name", "string"),
            ("value", "float"),
        )
        expect_equal(result, (("id", "int"), ("name", "string"), ("value", "float")))

    @staticmethod
    def test_schema_output_tuple_empty() -> None:
        """Test creating empty schema."""
        result = schema_output_tuple()
        expect_equal(result, ())


class TestColumnTypes:
    """Test suite for ColumnTypes constants."""

    @staticmethod
    def test_string_constant() -> None:
        """Test STRING constant."""
        expect_equal(ColumnTypes.STRING, "string")

    @staticmethod
    def test_int_constant() -> None:
        """Test INT constant."""
        expect_equal(ColumnTypes.INT, "int")

    @staticmethod
    def test_float_constant() -> None:
        """Test FLOAT constant."""
        expect_equal(ColumnTypes.FLOAT, "float")

    @staticmethod
    def test_bool_constant() -> None:
        """Test BOOL constant."""
        expect_equal(ColumnTypes.BOOL, "bool")

    @staticmethod
    def test_datetime_constant() -> None:
        """Test DATETIME constant."""
        expect_equal(ColumnTypes.DATETIME, "datetime")


@pytest.mark.parametrize(
    ("columns", "expected_count"),
    [
        pytest.param([], 0, id="empty"),
        pytest.param([("id", "int")], 1, id="single"),
        pytest.param(
            [("a", "int"), ("b", "string"), ("c", "float")],
            3,
            id="multiple",
        ),
    ],
)
def test_schema_from_columns_parametrized(
    columns: list[tuple[str, str]],
    expected_count: int,
) -> None:
    """Parametrized test for schema_from_columns."""
    result = schema_from_columns(columns)
    expect_length(result, expected_count)
