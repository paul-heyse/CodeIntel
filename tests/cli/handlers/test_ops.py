"""Tests for ops handlers following the unified handler pattern."""

from __future__ import annotations

import pytest

from codeintel.cli.core.result_types import (
    DatasetDescribeResult,
    DatasetListResult,
    DatasetVerifyResult,
    OperationCallResult,
    OperationListResult,
)
from codeintel.cli.handlers.ops import ServeStartResult, op_list_handler
from codeintel.serving.operations.catalog import iter_operations
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli_context import make_command_context


@pytest.mark.usefixtures("operation_registry_harness_fixture")
def test_op_list_handler_returns_ok() -> None:
    """Handler returns success with operation list."""
    with make_command_context({}, operation_id="ops.test") as ctx:
        result = op_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, OperationListResult)
    if result.data is not None:
        expect_equal(result.data.count, len(result.data.operations))


@pytest.mark.usefixtures("operation_registry_harness_fixture")
def test_op_list_handler_filters_by_category() -> None:
    """Handler filters operations by category."""
    operations = tuple(iter_operations())
    categorized = [op for op in operations if op.category]
    if not categorized:
        pytest.skip("No categorized operations available")

    target_category = categorized[0].category

    with make_command_context({"category": target_category}, operation_id="ops.test") as ctx:
        result = op_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_true(all(op["category"] == target_category for op in result.data.operations))


def test_op_list_result_to_dict() -> None:
    """OperationListResult.to_dict returns expected structure."""
    result = OperationListResult(
        operations=[{"id": "test", "category": "test"}],
        count=1,
    )

    data = result.to_dict()

    expect_equal(data["count"], 1)
    expect_equal(len(result.operations), 1)


def test_dataset_list_result_to_dict() -> None:
    """DatasetListResult.to_dict returns expected structure."""
    result = DatasetListResult(
        datasets=[{"name": "test", "table_key": "test.table"}],
        count=1,
    )

    data = result.to_dict()

    expect_equal(data["count"], 1)
    expect_equal(len(result.datasets), 1)


def test_dataset_describe_result_to_dict() -> None:
    """DatasetDescribeResult.to_dict returns expected structure."""
    result = DatasetDescribeResult(
        table_key="test.table",
        name="Test Dataset",
        description="A test dataset",
        owner_package="test_package",
        columns=[{"name": "id", "type": "INTEGER", "nullable": False}],
        row_count=100,
        upstream_dependencies=["other.table"],
    )

    data = result.to_dict()

    expect_equal(data["table_key"], "test.table")
    expect_equal(data["name"], "Test Dataset")
    expect_equal(data["owner_package"], "test_package")


def test_dataset_verify_result_to_dict() -> None:
    """DatasetVerifyResult.to_dict returns expected structure."""
    result = DatasetVerifyResult(
        verified=True,
        issues=[],
    )

    data = result.to_dict()

    expect_true(data["verified"])
    expect_equal(data["issues"], [])


def test_op_call_result_to_dict() -> None:
    """OperationCallResult.to_dict returns expected structure."""
    result = OperationCallResult(
        operation_id="test-op",
        result={"data": "value"},
    )

    data = result.to_dict()

    expect_equal(data["operation_id"], "test-op")
    expect_equal(data["result"], {"data": "value"})


def test_serve_start_result_to_dict() -> None:
    """ServeStartResult.to_dict returns expected structure."""
    result = ServeStartResult(
        server_type="http",
        host="127.0.0.1",
        port=8000,
        auto_pipeline=True,
        repo="test/repo",
        commit="abc123",
        db_path="/path/to/db",
    )

    data = result.to_dict()

    expect_equal(data["server_type"], "http")
    expect_equal(data["host"], "127.0.0.1")
    expect_equal(data["port"], 8000)
    expect_true(data["auto_pipeline"])
