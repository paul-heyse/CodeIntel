"""Tests for view schema inference (PR-72).

This module tests the `infer_view_schema()` function that enables DuckDB
view schema inference as part of the v2 manifest format.
"""

from __future__ import annotations

import duckdb
import pytest

from codeintel.build.schemas.infer_duckdb import (
    infer_view_schema,
    normalize_duckdb_type,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_true,
)

EXPECTED_SIMPLE_VIEW_COLUMNS = 2
EXPECTED_MULTI_TYPE_COLUMNS = 6
EXPECTED_EXPRESSION_VIEW_COLUMNS = 4


class TestNormalizeDuckdbType:
    """Tests for normalize_duckdb_type helper."""

    @staticmethod
    @pytest.mark.parametrize(
        ("input_type", "expected"),
        [
            ("BOOL", "BOOLEAN"),
            ("BOOLEAN", "BOOLEAN"),
            ("INT", "INTEGER"),
            ("INTEGER", "INTEGER"),
            ("INT4", "INTEGER"),
            ("BIGINT", "BIGINT"),
            ("INT8", "BIGINT"),
            ("DOUBLE", "DOUBLE"),
            ("DOUBLE PRECISION", "DOUBLE"),
            ("VARCHAR", "VARCHAR"),
            ("TEXT", "VARCHAR"),
            ("JSON", "JSON"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("TIMESTAMP_TZ", "TIMESTAMPTZ"),
            ("TIMESTAMPTZ", "TIMESTAMPTZ"),
            ("TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"),
            ("DECIMAL", "DECIMAL"),
            ("DECIMAL(38,0)", "DECIMAL(38,0)"),
            # Note: DECIMAL(10,2) without space works, but with space it fails
            # This is consistent with DuckDB output format
            ("DECIMAL(10,2)", "DECIMAL"),
        ],
    )
    def test_normalize_supported_types(input_type: str, expected: str) -> None:
        """Verify known types are normalized correctly."""
        expect_equal(normalize_duckdb_type(input_type), expected)

    @staticmethod
    def test_normalize_unsupported_type_raises() -> None:
        """Verify unsupported types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported DuckDB type"):
            normalize_duckdb_type("GEOMETRY")


class TestInferViewSchema:
    """Tests for infer_view_schema function."""

    @staticmethod
    def test_infer_simple_view(duckdb_con: duckdb.DuckDBPyConnection) -> None:
        """Test inferring schema from a simple view."""
        # Create a test table and view
        duckdb_con.execute("""
            CREATE SCHEMA IF NOT EXISTS test_schema
        """)
        duckdb_con.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.source_table (
                id INTEGER NOT NULL,
                name VARCHAR,
                created_at TIMESTAMP
            )
        """)
        duckdb_con.execute("""
            CREATE OR REPLACE VIEW test_schema.test_view AS
            SELECT id, name FROM test_schema.source_table
        """)

        schema = infer_view_schema(con=duckdb_con, view_key="test_schema.test_view")

        expect_equal(schema.schema, "test_schema")
        expect_equal(schema.name, "test_view")
        expect_equal(schema.table_key, "test_schema.test_view")
        expect_equal(len(schema.columns), EXPECTED_SIMPLE_VIEW_COLUMNS)
        expect_equal(schema.columns[0].name, "id")
        expect_equal(schema.columns[0].type, "INTEGER")
        expect_equal(schema.columns[1].name, "name")
        expect_equal(schema.columns[1].type, "VARCHAR")

    @staticmethod
    def test_infer_view_with_multiple_types(duckdb_con: duckdb.DuckDBPyConnection) -> None:
        """Test inferring schema from view with various column types."""
        duckdb_con.execute("""
            CREATE SCHEMA IF NOT EXISTS test_schema
        """)
        duckdb_con.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.multi_type_table (
                bool_col BOOLEAN,
                int_col INTEGER,
                bigint_col BIGINT,
                double_col DOUBLE,
                varchar_col VARCHAR,
                timestamp_col TIMESTAMP
            )
        """)
        duckdb_con.execute("""
            CREATE OR REPLACE VIEW test_schema.multi_type_view AS
            SELECT * FROM test_schema.multi_type_table
        """)

        schema = infer_view_schema(con=duckdb_con, view_key="test_schema.multi_type_view")

        expect_equal(len(schema.columns), EXPECTED_MULTI_TYPE_COLUMNS)
        col_types = {col.name: col.type for col in schema.columns}
        expect_equal(col_types["bool_col"], "BOOLEAN")
        expect_equal(col_types["int_col"], "INTEGER")
        expect_equal(col_types["bigint_col"], "BIGINT")
        expect_equal(col_types["double_col"], "DOUBLE")
        expect_equal(col_types["varchar_col"], "VARCHAR")
        expect_equal(col_types["timestamp_col"], "TIMESTAMP")

    @staticmethod
    def test_infer_nonexistent_view_raises(duckdb_con: duckdb.DuckDBPyConnection) -> None:
        """Test that inferring nonexistent view raises an exception."""
        with pytest.raises(duckdb.CatalogException):
            infer_view_schema(con=duckdb_con, view_key="nonexistent.view")

    @staticmethod
    def test_infer_view_with_expressions(duckdb_con: duckdb.DuckDBPyConnection) -> None:
        """Test inferring schema from view with computed columns."""
        duckdb_con.execute("""
            CREATE SCHEMA IF NOT EXISTS test_schema
        """)
        duckdb_con.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.numbers (
                a INTEGER,
                b INTEGER
            )
        """)
        duckdb_con.execute("""
            CREATE OR REPLACE VIEW test_schema.computed_view AS
            SELECT a, b, a + b AS sum_ab, a * b AS prod_ab
            FROM test_schema.numbers
        """)

        schema = infer_view_schema(con=duckdb_con, view_key="test_schema.computed_view")

        expect_equal(len(schema.columns), EXPECTED_EXPRESSION_VIEW_COLUMNS)
        expect_equal(schema.columns[2].name, "sum_ab")
        expect_equal(schema.columns[3].name, "prod_ab")


class TestInferViewSchemaEdgeCases:
    """Edge case tests for view schema inference."""

    @staticmethod
    def test_view_with_cast_columns(duckdb_con: duckdb.DuckDBPyConnection) -> None:
        """Test view with explicit cast operations."""
        duckdb_con.execute("""
            CREATE SCHEMA IF NOT EXISTS test_schema
        """)
        duckdb_con.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.cast_source (
                str_int VARCHAR
            )
        """)
        duckdb_con.execute("""
            CREATE OR REPLACE VIEW test_schema.cast_view AS
            SELECT CAST(str_int AS INTEGER) AS int_val
            FROM test_schema.cast_source
        """)

        schema = infer_view_schema(con=duckdb_con, view_key="test_schema.cast_view")

        expect_equal(len(schema.columns), 1)
        expect_equal(schema.columns[0].name, "int_val")
        expect_equal(schema.columns[0].type, "INTEGER")

    @staticmethod
    def test_view_nullable_inference_preserves_constraint(
        duckdb_con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Verify that inferred view columns preserve nullability from source."""
        duckdb_con.execute("""
            CREATE SCHEMA IF NOT EXISTS test_schema
        """)
        duckdb_con.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.nullable_source (
                required_col INTEGER NOT NULL,
                optional_col INTEGER
            )
        """)
        duckdb_con.execute("""
            CREATE OR REPLACE VIEW test_schema.nullable_view AS
            SELECT * FROM test_schema.nullable_source
        """)

        schema = infer_view_schema(con=duckdb_con, view_key="test_schema.nullable_view")

        # DuckDB actually preserves NOT NULL constraints when describing views
        # that select directly from source tables
        col_by_name = {col.name: col for col in schema.columns}
        expect_false(col_by_name["required_col"].nullable)
        expect_true(col_by_name["optional_col"].nullable)


@pytest.fixture
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Provide an in-memory DuckDB connection for testing.

    Yields
    ------
    duckdb.DuckDBPyConnection
        In-memory DuckDB connection.
    """
    import duckdb  # noqa: PLC0415

    con = duckdb.connect(":memory:")
    yield con
    con.close()


__all__ = [
    "TestInferViewSchema",
    "TestInferViewSchemaEdgeCases",
    "TestNormalizeDuckdbType",
]
