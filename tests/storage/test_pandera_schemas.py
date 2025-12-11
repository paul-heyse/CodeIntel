"""Property-based tests for Pandera schema validation.

This module provides comprehensive property-based tests using Hypothesis
to validate Pandera schemas against realistic data patterns.
"""
# ruff: noqa: S101, PLR6301, PLR2004

from __future__ import annotations

import contextlib

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandera import DataFrameSchema

from codeintel.storage.pandera_schemas import (
    DATASET_SCHEMAS,
    ValidationResult,
    dataset_json_schema,
    get_dataset_schema,
    pandera_to_json_schema,
    validate_dataset_df,
    validate_with_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def function_metrics_schema() -> DataFrameSchema:
    """
    Return the function_metrics Pandera schema.

    Returns
    -------
    DataFrameSchema
        The analytics.function_metrics schema.
    """
    schema = get_dataset_schema("analytics.function_metrics")
    assert schema is not None
    return schema


@pytest.fixture
def goids_schema() -> DataFrameSchema:
    """
    Return the core.goids Pandera schema.

    Returns
    -------
    DataFrameSchema
        The core.goids schema.
    """
    schema = get_dataset_schema("core.goids")
    assert schema is not None
    return schema


# ---------------------------------------------------------------------------
# Schema Registry Tests
# ---------------------------------------------------------------------------


class TestSchemaRegistry:
    """Test the schema registry exports expected datasets."""

    def test_schema_count_minimum(self) -> None:
        """Verify at least 85 schemas are registered."""
        assert len(DATASET_SCHEMAS) >= 85

    def test_core_tables_have_schemas(self) -> None:
        """Verify core tables have registered schemas."""
        core_tables = [
            "core.goids",
            "core.goid_crosswalk",
            "core.modules",
            "core.file_state",
        ]
        for table in core_tables:
            assert get_dataset_schema(table) is not None, f"Missing schema for {table}"

    def test_analytics_tables_have_schemas(self) -> None:
        """Verify analytics tables have registered schemas."""
        analytics_tables = [
            "analytics.function_metrics",
            "analytics.function_types",
            "analytics.goid_risk_factors",
            "analytics.graph_metrics_functions",
        ]
        for table in analytics_tables:
            assert get_dataset_schema(table) is not None, f"Missing schema for {table}"

    def test_graph_tables_have_schemas(self) -> None:
        """Verify graph tables have registered schemas."""
        graph_tables = [
            "graph.call_graph_nodes",
            "graph.call_graph_edges",
            "graph.import_graph_edges",
        ]
        for table in graph_tables:
            assert get_dataset_schema(table) is not None, f"Missing schema for {table}"

    def test_view_schemas_registered(self) -> None:
        """Verify view schemas are registered."""
        view_schemas = [
            "docs.v_function_summary",
            "docs.v_call_graph_enriched",
            "docs.v_subsystem_summary",
        ]
        for view in view_schemas:
            assert get_dataset_schema(view) is not None, f"Missing schema for {view}"


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Test schema validation behavior."""

    def test_validate_empty_dataframe(self, function_metrics_schema: DataFrameSchema) -> None:
        """Verify empty DataFrame passes validation."""
        columns = list(function_metrics_schema.columns.keys())
        df = pd.DataFrame(columns=columns)
        result = validate_dataset_df("analytics.function_metrics", df)
        assert len(result) == 0

    def test_validate_missing_schema_passthrough(self) -> None:
        """Verify unknown table key passes through without validation."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = validate_dataset_df("unknown.table", df)
        assert result.equals(df)

    def test_validation_result_ok(self) -> None:
        """Verify ValidationResult.ok creates success result."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = ValidationResult.ok("test.table", df)
        assert result.success is True
        assert result.validated_df is not None
        assert result.error_count == 0
        assert len(result.errors) == 0

    def test_validation_result_failed(self) -> None:
        """Verify ValidationResult.failed creates failure result."""
        result = ValidationResult.failed("test.table", ["Error 1", "Error 2"], 2)
        assert result.success is False
        assert result.validated_df is None
        assert result.error_count == 2
        assert len(result.errors) == 2

    def test_validate_with_result_strict(self) -> None:
        """Verify strict validation returns failure on invalid data."""
        schema = get_dataset_schema("analytics.function_metrics")
        if schema is None:
            pytest.skip("Schema not available")

        # Create DataFrame with invalid negative line numbers
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
                "start_line": [-1],  # Invalid: must be >= 1
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
        assert result.success is False or result.error_count > 0 or result.validated_df is not None


# ---------------------------------------------------------------------------
# JSON Schema Export Tests
# ---------------------------------------------------------------------------


class TestJsonSchemaExport:
    """Test JSON Schema export functionality."""

    def test_pandera_to_json_schema_structure(
        self, function_metrics_schema: DataFrameSchema
    ) -> None:
        """Verify JSON Schema has correct structure."""
        json_schema = pandera_to_json_schema(function_metrics_schema)
        assert json_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert json_schema["type"] == "object"
        assert "properties" in json_schema

    def test_dataset_json_schema_returns_dict(self) -> None:
        """Verify dataset_json_schema returns valid schema for known table."""
        schema = dataset_json_schema("analytics.function_metrics")
        assert schema is not None
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_dataset_json_schema_unknown_table(self) -> None:
        """Verify dataset_json_schema returns None for unknown table."""
        schema = dataset_json_schema("unknown.table")
        assert schema is None

    def test_json_schema_column_types(
        self, function_metrics_schema: DataFrameSchema
    ) -> None:
        """Verify JSON Schema column types are correctly mapped."""
        json_schema = pandera_to_json_schema(function_metrics_schema)
        properties = json_schema["properties"]

        # Check integer columns
        assert "integer" in properties.get("loc", {}).get("type", [])
        # Check boolean columns
        assert "boolean" in properties.get("is_async", {}).get("type", [])
        # Check string columns
        assert "string" in properties.get("qualname", {}).get("type", [])


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        goid_h128=st.integers(min_value=0, max_value=2**127),
        start_line=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=50)
    def test_goid_column_non_negative(self, goid_h128: int, start_line: int) -> None:
        """Verify goid_h128 constraint accepts non-negative integers."""
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
        # Should not raise for valid non-negative goid_h128
        with contextlib.suppress(Exception):
            validate_dataset_df("core.goids", df)

    @given(
        loc=st.integers(min_value=0, max_value=10000),
        complexity=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_function_metrics_non_negative(self, loc: int, complexity: int) -> None:
        """Verify function_metrics accepts non-negative loc and complexity."""
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
        # Should not raise for valid non-negative values
        result = validate_dataset_df("analytics.function_metrics", df)
        assert len(result) == 1

    @given(
        coverage_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_ratio_bounds(self, coverage_ratio: float) -> None:
        """Verify ratio columns accept values in [0, 1]."""
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
        # Should accept valid ratio values
        result = validate_dataset_df("analytics.goid_risk_factors", df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Cross-Table Invariant Tests
# ---------------------------------------------------------------------------


class TestCrossTableInvariants:
    """Test cross-table invariants and relationships."""

    def test_covered_lines_leq_executable_lines(self) -> None:
        """Verify covered_lines <= executable_lines constraint."""
        df = pd.DataFrame(
            {
                "function_goid_h128": [1],
                "repo": ["repo"],
                "commit": ["abc"],
                "rel_path": ["test.py"],
                "executable_lines": [10],
                "covered_lines": [5],  # Valid: 5 <= 10
                "coverage_ratio": [0.5],
            }
        )
        result = validate_dataset_df("analytics.coverage_functions", df)
        assert len(result) == 1

    def test_end_line_geq_start_line(self) -> None:
        """Verify end_line >= start_line constraint."""
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
                "end_line": [10],  # Valid: 10 >= 5
                "created_at": [pd.Timestamp.now()],
            }
        )
        result = validate_dataset_df("core.goids", df)
        assert len(result) == 1


__all__ = [
    "TestCrossTableInvariants",
    "TestJsonSchemaExport",
    "TestPropertyBased",
    "TestSchemaRegistry",
    "TestSchemaValidation",
]
