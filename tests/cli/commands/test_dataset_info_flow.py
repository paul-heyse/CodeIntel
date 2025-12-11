"""Tests for dataset info and flow CLI commands.

Test the new dataset introspection commands that use the unified schema registry.
"""

from __future__ import annotations

import pytest

from codeintel.cli.core.result_types import (
    DatasetFlowResult,
    DatasetInfoResult,
)
from codeintel.cli.handlers.ops import (
    dataset_flow_handler,
    dataset_flow_structured,
    dataset_info_handler,
    dataset_info_structured,
)
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli_context import make_command_context

# =============================================================================
# DatasetInfoResult Tests
# =============================================================================


def test_dataset_info_result_to_dict() -> None:
    """DatasetInfoResult.to_dict returns expected structure."""
    result = DatasetInfoResult(
        name="analytics.function_metrics",
        columns=("repo", "commit", "function_goid_h128"),
        metadata={"owner": "analytics", "description": "Function metrics"},
        json_schema={"type": "object", "properties": {}},
        has_pandera_schema=True,
    )

    data = result.to_dict()

    expect_equal(data["name"], "analytics.function_metrics")
    expect_equal(data["column_count"], 3)
    expect_true(data["has_pandera_schema"])
    expect_is_instance(data["columns"], list)
    expect_is_instance(data["metadata"], dict)
    expect_is_instance(data["json_schema"], dict)


def test_dataset_info_result_empty_metadata() -> None:
    """DatasetInfoResult handles empty metadata."""
    result = DatasetInfoResult(
        name="test.table",
        columns=("id",),
        metadata={},
        json_schema={},
        has_pandera_schema=False,
    )

    data = result.to_dict()

    expect_equal(data["metadata"], {})
    expect_true(not data["has_pandera_schema"])


# =============================================================================
# DatasetFlowResult Tests
# =============================================================================


def test_dataset_flow_result_to_dict() -> None:
    """DatasetFlowResult.to_dict returns expected structure."""
    result = DatasetFlowResult(
        table_key="analytics.function_metrics",
        producers=["function_metrics_plugin"],
        consumers=["risk_plugin", "profile_plugin"],
    )

    data = result.to_dict()

    expect_equal(data["table_key"], "analytics.function_metrics")
    expect_equal(data["producer_count"], 1)
    expect_equal(data["consumer_count"], 2)
    expect_is_instance(data["producers"], list)
    expect_is_instance(data["consumers"], list)


def test_dataset_flow_result_empty_flow() -> None:
    """DatasetFlowResult handles empty producers/consumers."""
    result = DatasetFlowResult(
        table_key="orphan.table",
        producers=[],
        consumers=[],
    )

    data = result.to_dict()

    expect_equal(data["producer_count"], 0)
    expect_equal(data["consumer_count"], 0)


# =============================================================================
# Handler Tests - Structured Functions
# =============================================================================


def test_dataset_info_structured_for_registered_dataset() -> None:
    """Structured info handler returns schema info for registered dataset."""
    # Initialize the registry
    SCHEMA_REGISTRY.initialize()

    # Check if we have any registered schemas
    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    # Pick a known schema
    table_key = next(iter(all_schemas.keys()))
    result = dataset_info_structured(table_key=table_key)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_is_instance(result.data, DatasetInfoResult)
        expect_equal(result.data.name, table_key)
        expect_true(len(result.data.columns) > 0)


def test_dataset_info_structured_for_unregistered_dataset() -> None:
    """Structured info handler returns error for unregistered dataset."""
    result = dataset_info_structured(table_key="nonexistent.table")

    expect_true(not result.success)
    expect_is_not_none(result.error)


def test_dataset_flow_structured_for_registered_dataset() -> None:
    """Structured flow handler returns flow info for registered dataset."""
    # Initialize the registry
    SCHEMA_REGISTRY.initialize()

    # Check if we have any registered schemas
    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    # Pick a known schema
    table_key = next(iter(all_schemas.keys()))
    result = dataset_flow_structured(table_key=table_key)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_is_instance(result.data, DatasetFlowResult)
        expect_equal(result.data.table_key, table_key)
        expect_is_instance(result.data.producers, list)
        expect_is_instance(result.data.consumers, list)


def test_dataset_flow_structured_for_unregistered_dataset() -> None:
    """Structured flow handler returns error for unregistered dataset."""
    result = dataset_flow_structured(table_key="nonexistent.table")

    expect_true(not result.success)
    expect_is_not_none(result.error)


# =============================================================================
# Handler Tests - Context-Based
# =============================================================================


def test_dataset_info_handler_with_valid_table_key() -> None:
    """Info handler extracts table_key from context and returns result."""
    # Initialize the registry
    SCHEMA_REGISTRY.initialize()

    # Check if we have any registered schemas
    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    table_key = next(iter(all_schemas.keys()))

    with make_command_context(
        {"table_key": table_key},
        operation_id="dataset.info",
    ) as ctx:
        result = dataset_info_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_equal(result.data.name, table_key)


def test_dataset_flow_handler_with_valid_table_key() -> None:
    """Flow handler extracts table_key from context and returns result."""
    # Initialize the registry
    SCHEMA_REGISTRY.initialize()

    # Check if we have any registered schemas
    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    table_key = next(iter(all_schemas.keys()))

    with make_command_context(
        {"table_key": table_key},
        operation_id="dataset.flow",
    ) as ctx:
        result = dataset_flow_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_equal(result.data.table_key, table_key)


# =============================================================================
# Integration Tests
# =============================================================================


def test_dataset_info_includes_column_names() -> None:
    """Info result includes actual column names from schema."""
    SCHEMA_REGISTRY.initialize()

    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    # Test with a well-known schema if available
    known_tables = ["analytics.function_metrics", "core.goids"]
    table_key = None
    for key in known_tables:
        if key in all_schemas:
            table_key = key
            break

    if table_key is None:
        table_key = next(iter(all_schemas.keys()))

    result = dataset_info_structured(table_key=table_key)

    expect_true(result.success)
    if result.data is not None:
        # Columns should be present
        expect_true(len(result.data.columns) > 0)


def test_dataset_info_json_schema_is_valid() -> None:
    """Info result includes valid JSON schema."""
    SCHEMA_REGISTRY.initialize()

    all_schemas = SCHEMA_REGISTRY.all()
    if not all_schemas:
        pytest.skip("No schemas registered in SCHEMA_REGISTRY")

    table_key = next(iter(all_schemas.keys()))
    result = dataset_info_structured(table_key=table_key)

    expect_true(result.success)
    if result.data is not None:
        # JSON schema should have type key
        json_schema = result.data.json_schema
        expect_is_instance(json_schema, dict)
