"""Schema parity checks for analytics row models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import get_type_hints

import pytest

from codeintel.config.datasets import (
    DATASET_CONTRACTS_BY_TABLE_KEY,
    BehavioralCoverageRowModel,
    FunctionMetricsRow,
    FunctionTypesRow,
    GraphMetricsFunctionsExtRow,
    GraphMetricsFunctionsRow,
    GraphMetricsModulesExtRow,
    GraphMetricsModulesRow,
    ProfileRowModel,
)


def _assert_row_matches_table(row_type: type[Mapping[str, object]], table_key: str) -> None:
    """Verify TypedDict annotations align with the DatasetContract schema."""
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    if contract is None or contract.schema is None:
        pytest.fail(f"{table_key} has no contract schema")
        return
    expected_cols = [col.name for col in contract.schema.columns]
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
    _assert_row_matches_table(ProfileRowModel, "analytics.test_profile")


def test_behavioral_coverage_row_matches_schema() -> None:
    """behavioral_coverage row fields should match table schema."""
    _assert_row_matches_table(BehavioralCoverageRowModel, "analytics.behavioral_coverage")


def test_graph_metrics_functions_row_matches_schema() -> None:
    """graph_metrics_functions row fields should match table schema."""
    _assert_row_matches_table(
        GraphMetricsFunctionsRow,
        "analytics.graph_metrics_functions",
    )


def test_graph_metrics_modules_row_matches_schema() -> None:
    """graph_metrics_modules row fields should match table schema."""
    _assert_row_matches_table(
        GraphMetricsModulesRow,
        "analytics.graph_metrics_modules",
    )


def test_graph_metrics_functions_ext_row_matches_schema() -> None:
    """graph_metrics_functions_ext row fields should match table schema."""
    _assert_row_matches_table(
        GraphMetricsFunctionsExtRow,
        "analytics.graph_metrics_functions_ext",
    )


def test_graph_metrics_modules_ext_row_matches_schema() -> None:
    """graph_metrics_modules_ext row fields should match table schema."""
    _assert_row_matches_table(
        GraphMetricsModulesExtRow,
        "analytics.graph_metrics_modules_ext",
    )
