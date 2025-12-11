"""Tests for ops handlers following the unified handler pattern."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from codeintel.cli.context import CommandContext, CommandContextBuilder
from codeintel.cli.core.result_types import (
    DatasetDescribeResult,
    DatasetListResult,
    DatasetVerifyResult,
    OperationCallResult,
    OperationListResult,
)
from codeintel.cli.handlers.ops import ServeStartResult, op_list_handler
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)


def test_op_list_handler_returns_ok() -> None:
    """Handler returns success with operation list."""
    mock_op = MagicMock()
    mock_op.id = "test-op"
    mock_op.category = "test"
    mock_op.summary = "Test operation"
    mock_op.http_path = "/api/test"
    mock_op.tool_name = "test_tool"

    with (
        _build_test_context(params={}) as ctx,
        patch(
            "codeintel.cli.handlers.ops.iter_operations",
            return_value=[mock_op],
        ),
    ):
        result = op_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, OperationListResult)
    if result.data is not None:
        expect_equal(result.data.count, 1)


def test_op_list_handler_filters_by_category() -> None:
    """Handler filters operations by category."""
    mock_op1 = MagicMock()
    mock_op1.id = "test-op"
    mock_op1.category = "test"
    mock_op1.summary = "Test operation"
    mock_op1.http_path = "/api/test"
    mock_op1.tool_name = "test_tool"

    mock_op2 = MagicMock()
    mock_op2.id = "other-op"
    mock_op2.category = "other"
    mock_op2.summary = "Other operation"
    mock_op2.http_path = "/api/other"
    mock_op2.tool_name = "other_tool"

    with (
        _build_test_context(params={"category": "test"}) as ctx,
        patch(
            "codeintel.cli.handlers.ops.iter_operations",
            return_value=[mock_op1, mock_op2],
        ),
    ):
        result = op_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_equal(result.data.count, 1)


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


@contextmanager
def _build_test_context(params: dict[str, object]) -> Iterator[CommandContext]:
    """Build a test context with mocked dependencies.

    Yields
    ------
    CommandContext
        Context configured with a mocked storage gateway.
    """
    mock_gateway = MagicMock(spec=StorageGateway)
    builder = (
        CommandContextBuilder()
        .with_params(params)
        .with_operation_id("ops.test")
        .with_injected_gateway(mock_gateway)
    )
    with ExitStack() as stack:
        ctx = stack.enter_context(builder.build())
        yield ctx
