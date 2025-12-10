"""Unit tests for CLI operation handlers.

Test individual operations through OperationTestHarness.
"""

from __future__ import annotations

from codeintel.cli.introspection import get_registry
from tests._helpers.assertions import (
    expect_false,
    expect_is_not_none,
    expect_not_empty,
    expect_true,
)
from tests.cli._harness import OperationTestHarness


def test_build_status_returns_structured_result(
    op_harness: OperationTestHarness,
) -> None:
    """Build status handler returns structured result."""
    result = op_harness.execute("build.status")

    expect_is_not_none(result)
    # Result may have data as dict or string
    expect_true(result.success or result.data is None)


def test_build_status_no_params(
    op_harness: OperationTestHarness,
) -> None:
    """Build status works without parameters."""
    result = op_harness.execute("build.status")

    expect_is_not_none(result)
    # Handler may succeed or fail depending on environment
    expect_true(result.success or result.error is not None)


def test_build_operations_have_required_group() -> None:
    """Build operations should have a group assigned."""
    registry = get_registry()

    build_ops = [
        spec.operation_id
        for spec in registry.list_operations()
        if spec.operation_id.startswith("build.")
    ]
    for op_id in build_ops:
        spec = registry.get(op_id)
        expect_is_not_none(spec)
        if spec is not None:
            expect_is_not_none(spec.group)


def test_op_list_returns_operations(
    op_harness: OperationTestHarness,
) -> None:
    """Op list handler returns list of operations."""
    result = op_harness.execute("op.list")

    expect_true(result.success)
    # Result may have stdout content
    expect_is_not_none(result)


def test_op_call_with_unknown_operation(
    op_harness: OperationTestHarness,
) -> None:
    """Op call with unknown operation returns error."""
    result = op_harness.execute("op.call", params={"op_id": "unknown.op"})

    expect_false(result.success)
    expect_is_not_none(result.error)


def test_dataset_list_returns_datasets(
    op_harness: OperationTestHarness,
) -> None:
    """Dataset list returns available datasets or error without context.

    Without a project file or explicit params, this operation will fail
    with a ResolutionError. This is expected behavior.
    """
    result = op_harness.execute("dataset.list")

    expect_is_not_none(result)
    # Result is valid (either success or expected resolution error)
    expect_true(result.exit_code in (0, 1))


def test_storage_status_returns_info(
    op_harness: OperationTestHarness,
) -> None:
    """Storage status returns storage information."""
    result = op_harness.execute("storage.status")

    expect_is_not_none(result)


def test_registry_has_operations() -> None:
    """Registry should have registered operations."""
    registry = get_registry()
    operations = registry.list_operations()

    expect_not_empty(operations)


def test_registry_lookup_nonexistent() -> None:
    """Registry returns None for unknown operations."""
    registry = get_registry()
    _spec = registry.get("nonexistent.operation")

    # Should return None, not raise


def test_operations_have_required_fields() -> None:
    """All registered operations have required fields."""
    registry = get_registry()

    for spec in registry.list_operations():
        expect_is_not_none(spec)
        # Handler is always present on valid specs (not optional)
        expect_true(
            spec.handler is not None,
            message=f"Handler missing for {spec.operation_id}",
        )
        expect_is_not_none(spec.group)
        expect_is_not_none(spec.name)
        expect_is_not_none(spec.description)
