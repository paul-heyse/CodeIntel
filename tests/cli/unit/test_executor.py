"""Unit tests for UnifiedOperationExecutor."""

from __future__ import annotations

import time

import pytest

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.cli_validation import StringValidator, ValidationSchema
from codeintel.cli.execution import (
    ExecutionContext,
    ExecutionResult,
    OperationCategory,
    OperationSpec,
    UnifiedOperationExecutor,
)
from codeintel.cli.results import CliResult
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

# Test constants
ESTIMATED_DURATION = 10.0
TIMEOUT_SECONDS = 30.0


def _success_handler(**kwargs: object) -> CliResult[dict[str, object]]:
    """Create a successful CLI result with provided kwargs.

    Returns
    -------
    CliResult[dict[str, object]]
        Successful result with parameters.
    """
    return CliResult.ok({"params": kwargs})


def _error_handler(**kwargs: object) -> CliResult[dict[str, object]]:
    """Create a failed CLI result with test error.

    Returns
    -------
    CliResult[dict[str, object]]
        Failed result with error details.
    """
    _ = kwargs  # Unused
    return CliResult.fail(
        ProblemDetail(
            type="urn:test:error",
            title="Test Error",
            detail="Handler error",
            status=500,
        )
    )


def _raising_handler(**kwargs: object) -> CliResult[dict[str, object]]:
    """Raise a RuntimeError for testing exception handling.

    Raises
    ------
    RuntimeError
        Always raises with test message.
    """
    _ = kwargs  # Unused
    msg = "Handler exception"
    raise RuntimeError(msg)


def test_spec_defaults() -> None:
    """Test default values for OperationSpec."""
    spec = OperationSpec(
        operation_id="test.op",
        handler=_success_handler,
    )

    expect_equal(spec.operation_id, "test.op")
    expect_equal(spec.category, OperationCategory.READ)
    expect_is_none(spec.param_schema)
    expect_false(spec.requires_progress)
    expect_false(spec.retryable)
    expect_false(bool(spec.description))


def test_spec_with_all_options() -> None:
    """Test OperationSpec with all options set."""
    schema = ValidationSchema()
    spec = OperationSpec(
        operation_id="test.compute",
        handler=_success_handler,
        category=OperationCategory.COMPUTE,
        param_schema=schema,
        requires_progress=True,
        estimated_duration=ESTIMATED_DURATION,
        retryable=True,
        timeout=TIMEOUT_SECONDS,
        description="A test operation",
    )

    expect_equal(spec.operation_id, "test.compute")
    expect_equal(spec.category, OperationCategory.COMPUTE)
    expect_equal(spec.param_schema, schema)
    expect_true(spec.requires_progress)
    expect_equal(spec.estimated_duration, ESTIMATED_DURATION)
    expect_true(spec.retryable)
    expect_equal(spec.timeout, TIMEOUT_SECONDS)
    expect_equal(spec.description, "A test operation")


def test_context_creation() -> None:
    """Test ExecutionContext creation."""
    ctx = ExecutionContext(
        operation_id="test.op",
        params={"key": "value"},
        output_format=OutputFormat.TEXT,
    )

    expect_equal(ctx.operation_id, "test.op")
    expect_equal(ctx.params, {"key": "value"})
    expect_equal(ctx.output_format, OutputFormat.TEXT)


def test_elapsed_seconds() -> None:
    """Test elapsed_seconds property."""
    ctx = ExecutionContext(
        operation_id="test.op",
        params={},
        output_format=OutputFormat.TEXT,
    )

    # Wait a small amount
    time.sleep(0.01)

    # Elapsed should be greater than 0
    expect_true(ctx.elapsed_seconds > 0, message="elapsed_seconds should be > 0")


def test_execute_success() -> None:
    """Test successful operation execution."""
    spec: OperationSpec[dict[str, object]] = OperationSpec(
        operation_id="test.success",
        handler=_success_handler,
        category=OperationCategory.READ,
    )
    executor = UnifiedOperationExecutor()

    result = executor.execute(spec, {"key": "value"}, render=False)

    expect_true(result.result.success)
    expect_is_not_none(result.result.data)
    data = result.result.data
    expect_is_not_none(data)
    if data is not None:
        params = data.get("params")
        expect_is_not_none(params)
        if isinstance(params, dict):
            expect_equal(params.get("key"), "value")
    expect_true(result.duration_seconds > 0, message="duration_seconds should be > 0")


def test_execute_handler_error() -> None:
    """Test operation that returns error result."""
    spec: OperationSpec[dict[str, object]] = OperationSpec(
        operation_id="test.error",
        handler=_error_handler,
        category=OperationCategory.READ,
    )
    executor = UnifiedOperationExecutor()

    result = executor.execute(spec, {}, render=False)

    expect_false(result.result.success)
    expect_is_not_none(result.result.error)
    if result.result.error is not None:
        expect_equal(result.result.error.title, "Test Error")


def test_execute_raises_exception() -> None:
    """Test operation that raises an exception."""
    spec: OperationSpec[dict[str, object]] = OperationSpec(
        operation_id="test.raise",
        handler=_raising_handler,
        category=OperationCategory.READ,
    )
    executor = UnifiedOperationExecutor()

    with pytest.raises(RuntimeError, match="Handler exception"):
        executor.execute(spec, {}, render=False)


def test_execute_with_validation_success() -> None:
    """Test operation with successful parameter validation."""
    schema = ValidationSchema()
    schema.add("name", StringValidator(min_length=1))

    spec: OperationSpec[dict[str, object]] = OperationSpec(
        operation_id="test.validated",
        handler=_success_handler,
        category=OperationCategory.READ,
        param_schema=schema,
    )
    executor = UnifiedOperationExecutor()

    result = executor.execute(spec, {"name": "test"}, render=False)

    expect_true(result.result.success)
    expect_equal(len(result.validation_errors), 0)


def test_execute_validation_failure() -> None:
    """Test operation with failing validation."""
    schema = ValidationSchema()
    schema.add("name", StringValidator(min_length=1))

    spec: OperationSpec[dict[str, object]] = OperationSpec(
        operation_id="test.validated",
        handler=_success_handler,
        category=OperationCategory.READ,
        param_schema=schema,
    )
    executor = UnifiedOperationExecutor()

    # Missing required param
    result = executor.execute(spec, {}, render=False)

    expect_false(result.result.success)
    expect_true(len(result.validation_errors) > 0, message="Should have validation errors")


def test_execute_different_categories() -> None:
    """Test operations with different categories."""
    categories = [
        OperationCategory.READ,
        OperationCategory.WRITE,
        OperationCategory.COMPUTE,
        OperationCategory.NETWORK,
        OperationCategory.BUILD,
    ]

    executor = UnifiedOperationExecutor()

    for category in categories:
        spec: OperationSpec[dict[str, object]] = OperationSpec(
            operation_id=f"test.{category.value}",
            handler=_success_handler,
            category=category,
        )
        result = executor.execute(spec, {}, render=False)
        expect_true(result.result.success, message=f"Failed for category {category}")


DURATION_VALUE = 1.5
RETRIES_VALUE = 2


def test_result_structure() -> None:
    """Test ExecutionResult structure."""
    cli_result: CliResult[str] = CliResult.ok("test")
    exec_result: ExecutionResult[str] = ExecutionResult(
        result=cli_result,
        duration_seconds=DURATION_VALUE,
        validation_errors=["error1"],
        retries=RETRIES_VALUE,
    )

    expect_true(exec_result.result.success)
    expect_equal(exec_result.result.data, "test")
    expect_equal(exec_result.duration_seconds, DURATION_VALUE)
    expect_equal(exec_result.validation_errors, ["error1"])
    expect_equal(exec_result.retries, RETRIES_VALUE)


def test_result_defaults() -> None:
    """Test ExecutionResult defaults."""
    cli_result: CliResult[str] = CliResult.ok("test")
    exec_result: ExecutionResult[str] = ExecutionResult(
        result=cli_result,
        duration_seconds=0.5,
    )

    expect_equal(exec_result.validation_errors, [])
    expect_equal(exec_result.retries, 0)
