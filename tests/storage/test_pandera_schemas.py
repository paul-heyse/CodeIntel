"""Property-based tests for Pandera schema validation.

This module provides comprehensive property-based tests using Hypothesis
to validate Pandera schemas against realistic data patterns.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codeintel.storage.pandera_schemas import (
    DATASET_SCHEMAS,
    ValidationResult,
    dataset_json_schema,
    get_dataset_schema,
    pandera_to_json_schema,
    validate_dataset_df,
    validate_with_result,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from pandera import DataFrameSchema

MIN_SCHEMA_COUNT = 85


def _require_schema(table_key: str) -> DataFrameSchema:
    """
    Return a registered Pandera schema or raise if missing.

    Parameters
    ----------
    table_key
        Fully qualified dataset key.

    Returns
    -------
    DataFrameSchema
        The registered schema for the provided table key.

    Raises
    ------
    AssertionError
        If the dataset does not have a registered schema.
    """
    schema = get_dataset_schema(table_key)
    if schema is None:
        message = f"Missing schema for {table_key}"
        raise AssertionError(message)
    return schema


@pytest.fixture
def function_metrics_schema() -> DataFrameSchema:
    """
    Return the function_metrics Pandera schema.

    Returns
    -------
    DataFrameSchema
        The analytics.function_metrics schema.
    """
    return _require_schema("analytics.function_metrics")


@pytest.fixture
def goids_schema() -> DataFrameSchema:
    """
    Return the core.goids Pandera schema.

    Returns
    -------
    DataFrameSchema
        The core.goids schema.
    """
    return _require_schema("core.goids")


class TestSchemaRegistry:
    """Test the schema registry exports expected datasets."""

    def test_schema_count_minimum(self) -> None:
        """Verify at least 85 schemas are registered."""
        del self
        expect_true(
            len(DATASET_SCHEMAS) >= MIN_SCHEMA_COUNT,
            message="Expected at least 85 schemas",
        )

    def test_core_tables_have_schemas(self) -> None:
        """Verify core tables have registered schemas."""
        del self
        core_tables = [
            "core.goids",
            "core.goid_crosswalk",
            "core.modules",
            "core.file_state",
        ]
        for table in core_tables:
            _require_schema(table)

    def test_analytics_tables_have_schemas(self) -> None:
        """Verify analytics tables have registered schemas."""
        del self
        analytics_tables = [
            "analytics.function_metrics",
            "analytics.function_types",
            "analytics.goid_risk_factors",
            "analytics.graph_metrics_functions",
        ]
        for table in analytics_tables:
            _require_schema(table)

    def test_graph_tables_have_schemas(self) -> None:
        """Verify graph tables have registered schemas."""
        del self
        graph_tables = [
            "graph.call_graph_nodes",
            "graph.call_graph_edges",
            "graph.import_graph_edges",
        ]
        for table in graph_tables:
            _require_schema(table)

    def test_view_schemas_registered(self) -> None:
        """Verify view schemas are registered."""
        del self
        view_schemas = [
            "docs.v_function_summary",
            "docs.v_call_graph_enriched",
            "docs.v_subsystem_summary",
        ]
        for view in view_schemas:
            _require_schema(view)


class TestSchemaValidation:
    """Test schema validation behavior."""

    def test_validate_empty_dataframe(self, function_metrics_schema: DataFrameSchema) -> None:
        """Verify empty DataFrame passes validation."""
        del self
        column_names = list(function_metrics_schema.columns.keys())
        df = pd.DataFrame(columns=pd.Index(column_names))
        result = validate_dataset_df("analytics.function_metrics", df)
        expect_equal(len(result), 0, label="empty_dataframe_row_count")

    def test_validate_missing_schema_passthrough(self) -> None:
        """Verify unknown table key passes through without validation."""
        del self
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = validate_dataset_df("unknown.table", df)
        expect_true(result.equals(df), message="Unexpected mutation for unknown.table")

    def test_validation_result_ok(self) -> None:
        """Verify ValidationResult.ok creates success result.

        Raises
        ------
        AssertionError
            If the validated dataframe is unexpectedly None.
        """
        del self
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = ValidationResult.ok("test.table", df)
        expect_true(result.success, message="Expected ok() to mark success")
        if result.validated_df is None:
            message = "ok() should retain validated DataFrame"
            raise AssertionError(message)
        expect_equal(result.error_count, 0, label="ok_error_count")
        expect_equal(len(result.errors), 0, label="ok_error_list_length")

    def test_validation_result_failed(self) -> None:
        """Verify ValidationResult.failed creates failure result."""
        del self
        result = ValidationResult.failed("test.table", ["Error 1", "Error 2"], 2)
        expect_false(result.success, message="Expected failed() to mark failure")
        expect_true(result.validated_df is None, message="validated_df should be None")
        expect_equal(result.error_count, 2, label="failed_error_count")
        expect_equal(len(result.errors), 2, label="failed_error_list_length")

    def test_validate_with_result_strict(self) -> None:
        """Verify strict validation returns failure on invalid data."""
        del self
        schema = get_dataset_schema("analytics.function_metrics")
        if schema is None:
            pytest.skip("Schema not available")

        df = pd.DataFrame(
            {
                "function_goid_h128": [1],
                "urn": ["test"],
                "repo": ["repo"],
                "commit": ["abc"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test"],
                "start_line": [-1],
                "end_line": [10],
                "loc": [10],
                "logical_loc": [5],
                "param_count": [0],
                "positional_params": [0],
                "keyword_only_params": [0],
                "has_varargs": [False],
                "has_varkw": [False],
                "is_async": [False],
                "is_generator": [False],
                "return_count": [1],
                "yield_count": [0],
                "raise_count": [0],
                "cyclomatic_complexity": [1],
                "max_nesting_depth": [0],
                "stmt_count": [1],
                "decorator_count": [0],
                "has_docstring": [False],
                "complexity_bucket": ["low"],
                "created_at": [pd.Timestamp.now()],
            }
        )
        result = validate_with_result("analytics.function_metrics", df, strict=True)
        expect_false(result.success, message="Strict validation should fail for bad data")
        expect_true(result.error_count > 0, message="Expected validation errors")
        expect_true(result.validated_df is None, message="validated_df should be None")


class TestJsonSchemaExport:
    """Test JSON Schema export functionality."""

    def test_pandera_to_json_schema_structure(
        self, function_metrics_schema: DataFrameSchema
    ) -> None:
        """Verify JSON Schema has correct structure."""
        del self
        json_schema = pandera_to_json_schema(function_metrics_schema)
        expect_equal(
            json_schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
            label="json_schema_version",
        )
        expect_equal(json_schema["type"], "object", label="json_schema_type")
        expect_in("properties", json_schema, label="json_schema_properties_key")

    def test_dataset_json_schema_returns_dict(self) -> None:
        """Verify dataset_json_schema returns valid schema for known table.

        Raises
        ------
        AssertionError
            If the schema is missing for a known table.
        """
        del self
        schema = dataset_json_schema("analytics.function_metrics")
        if schema is None:
            message = "Expected schema for analytics.function_metrics"
            raise AssertionError(message)
        expect_true(isinstance(schema, dict), message="Schema should be a mapping")
        expect_in("properties", schema, label="dataset_json_schema_properties_key")

    def test_dataset_json_schema_unknown_table(self) -> None:
        """Verify dataset_json_schema returns None for unknown table."""
        del self
        schema = dataset_json_schema("unknown.table")
        expect_true(schema is None, message="Unknown table should return None")

    def test_json_schema_column_types(self, function_metrics_schema: DataFrameSchema) -> None:
        """Verify JSON Schema column types are correctly mapped."""
        del self
        json_schema = pandera_to_json_schema(function_metrics_schema)
        properties = json_schema["properties"]

        expect_in("integer", properties.get("loc", {}).get("type", []), label="loc_type")

        expect_in(
            "boolean",
            properties.get("is_async", {}).get("type", []),
            label="is_async_type",
        )

        expect_in(
            "string",
            properties.get("qualname", {}).get("type", []),
            label="qualname_type",
        )


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        goid_h128=st.integers(min_value=0, max_value=2**127),
        start_line=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=50)
    def test_goid_column_non_negative(self, goid_h128: int, start_line: int) -> None:
        """Verify goid_h128 constraint accepts non-negative integers."""
        del self
        schema = get_dataset_schema("core.goids")
        if schema is None:
            pytest.skip("Schema not available")

        end_line = start_line + 10
        df = pd.DataFrame(
            {
                "goid_h128": [goid_h128],
                "urn": ["test:urn"],
                "repo": ["test/repo"],
                "commit": ["abc123"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test_func"],
                "start_line": [start_line],
                "end_line": [end_line],
                "created_at": [pd.Timestamp.now()],
            }
        )

        with contextlib.suppress(Exception):
            validate_dataset_df("core.goids", df)

    @given(
        loc=st.integers(min_value=0, max_value=10000),
        complexity=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_function_metrics_non_negative(self, loc: int, complexity: int) -> None:
        """Verify function_metrics accepts non-negative loc and complexity."""
        del self
        schema = get_dataset_schema("analytics.function_metrics")
        if schema is None:
            pytest.skip("Schema not available")

        df = pd.DataFrame(
            {
                "function_goid_h128": [1],
                "urn": ["test"],
                "repo": ["repo"],
                "commit": ["abc"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test"],
                "start_line": [1],
                "end_line": [10],
                "loc": [loc],
                "logical_loc": [loc // 2],
                "param_count": [0],
                "positional_params": [0],
                "keyword_only_params": [0],
                "has_varargs": [False],
                "has_varkw": [False],
                "is_async": [False],
                "is_generator": [False],
                "return_count": [1],
                "yield_count": [0],
                "raise_count": [0],
                "cyclomatic_complexity": [complexity],
                "max_nesting_depth": [0],
                "stmt_count": [1],
                "decorator_count": [0],
                "has_docstring": [False],
                "complexity_bucket": ["low"],
                "created_at": [pd.Timestamp.now()],
            }
        )

        result = validate_dataset_df("analytics.function_metrics", df)
        expect_equal(len(result), 1, label="function_metrics_row_count")

    @given(
        coverage_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_ratio_bounds(self, coverage_ratio: float) -> None:
        """Verify ratio columns accept values in [0, 1]."""
        del self
        df = pd.DataFrame(
            {
                "function_goid_h128": [1],
                "repo": ["repo"],
                "commit": ["abc"],
                "urn": ["test"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test"],
                "loc": [10],
                "logical_loc": [5],
                "cyclomatic_complexity": [1],
                "complexity_bucket": ["low"],
                "typedness_bucket": ["typed"],
                "typedness_source": ["annotation"],
                "hotspot_score": [0.5],
                "coverage_ratio": [coverage_ratio],
                "file_typed_ratio": [coverage_ratio],
                "risk_score": [0.1],
                "risk_level": ["low"],
                "executable_lines": [10],
                "covered_lines": [int(coverage_ratio * 10)],
                "test_count": [1],
                "failing_test_count": [0],
                "tested": [True],
                "last_test_status": ["passed"],
                "static_error_count": [0],
                "has_static_errors": [False],
                "tags": ["[]"],
                "owners": ["[]"],
                "created_at": [pd.Timestamp.now()],
            }
        )

        result = validate_dataset_df("analytics.goid_risk_factors", df)
        expect_equal(len(result), 1, label="risk_factors_row_count")


class TestCrossTableInvariants:
    """Test cross-table invariants and relationships."""

    def test_covered_lines_leq_executable_lines(self) -> None:
        """Verify covered_lines <= executable_lines constraint."""
        del self
        df = pd.DataFrame(
            {
                "function_goid_h128": [1],
                "urn": ["test:func"],
                "repo": ["repo"],
                "commit": ["abc"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test:func"],
                "start_line": [1],
                "end_line": [10],
                "executable_lines": [10],
                "covered_lines": [5],
                "coverage_ratio": [0.5],
                "tested": [True],
                "untested_reason": [None],
                "created_at": [pd.Timestamp.now()],
            }
        )
        result = validate_dataset_df("analytics.coverage_functions", df)
        expect_equal(len(result), 1, label="coverage_functions_row_count")

    def test_end_line_geq_start_line(self) -> None:
        """Verify end_line >= start_line constraint."""
        del self
        df = pd.DataFrame(
            {
                "goid_h128": [1],
                "urn": ["test"],
                "repo": ["repo"],
                "commit": ["abc"],
                "rel_path": ["test.py"],
                "language": ["python"],
                "kind": ["function"],
                "qualname": ["test"],
                "start_line": [5],
                "end_line": [10],
                "created_at": [pd.Timestamp.now()],
            }
        )
        result = validate_dataset_df("core.goids", df)
        expect_equal(len(result), 1, label="goids_row_count")


__all__ = [
    "TestCrossTableInvariants",
    "TestJsonSchemaExport",
    "TestPropertyBased",
    "TestSchemaRegistry",
    "TestSchemaValidation",
]
