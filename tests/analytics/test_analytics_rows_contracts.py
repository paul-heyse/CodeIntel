"""Schema parity checks for analytics row models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import get_type_hints

import pytest

from codeintel.analytics.rows.function_metrics import FunctionMetricsRow
from codeintel.analytics.rows.function_types import FunctionTypesRow
from codeintel.analytics.rows.graph_metrics import (
    FunctionGraphMetricsRow,
    ModuleGraphMetricsRow,
)
from codeintel.analytics.rows.graph_metrics_ext import (
    FunctionGraphMetricsExtRow,
    ModuleGraphMetricsExtRow,
)
from codeintel.analytics.rows.test_profiles import BehavioralCoverageRow, TestProfileRow
from codeintel.config.schemas.tables import TABLE_SCHEMAS


def _assert_row_matches_table(row_type: type[Mapping[str, object]], table_key: str) -> None:
    """Verify TypedDict annotations align with the registered table schema."""
    schema = TABLE_SCHEMAS[table_key]
    expected_cols = [col.name for col in schema.columns]
    annotations = get_type_hints(row_type)
    actual_cols = list(annotations.keys())
    if actual_cols != expected_cols:
        pytest.fail(f"{table_key} mismatch: {actual_cols} != {expected_cols}")


def test_function_metrics_row_matches_schema() -> None:
    """function_metrics row fields should match table schema."""
    _assert_row_matches_table(FunctionMetricsRow, "analytics.function_metrics")


def test_function_types_row_matches_schema() -> None:
    """function_types row fields should match table schema."""
    _assert_row_matches_table(FunctionTypesRow, "analytics.function_types")


def test_test_profile_row_matches_schema() -> None:
    """test_profile row fields should match table schema."""
    _assert_row_matches_table(TestProfileRow, "analytics.test_profile")


def test_behavioral_coverage_row_matches_schema() -> None:
    """behavioral_coverage row fields should match table schema."""
    _assert_row_matches_table(BehavioralCoverageRow, "analytics.behavioral_coverage")


def test_graph_metrics_functions_row_matches_schema() -> None:
    """graph_metrics_functions row fields should match table schema."""
    _assert_row_matches_table(
        FunctionGraphMetricsRow,
        "analytics.graph_metrics_functions",
    )


def test_graph_metrics_modules_row_matches_schema() -> None:
    """graph_metrics_modules row fields should match table schema."""
    _assert_row_matches_table(
        ModuleGraphMetricsRow,
        "analytics.graph_metrics_modules",
    )


def test_graph_metrics_functions_ext_row_matches_schema() -> None:
    """graph_metrics_functions_ext row fields should match table schema."""
    _assert_row_matches_table(
        FunctionGraphMetricsExtRow,
        "analytics.graph_metrics_functions_ext",
    )


def test_graph_metrics_modules_ext_row_matches_schema() -> None:
    """graph_metrics_modules_ext row fields should match table schema."""
    _assert_row_matches_table(
        ModuleGraphMetricsExtRow,
        "analytics.graph_metrics_modules_ext",
    )
