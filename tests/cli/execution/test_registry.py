"""Tests for unified operation registry.

Test the OperationRegistry and OperationSpec classes from the new
unified registry in execution/registry.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pytest

from codeintel.cli.core import CliResult
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------
EXPECTED_COUNT_ZERO = 0
EXPECTED_COUNT_ONE = 1
EXPECTED_COUNT_TWO = 2
EXPECTED_COUNT_FOUR = 4
EXPECTED_COUNT_FIVE = 5


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


class _StrContainer(Protocol):
    def __contains__(self, item: str, /) -> bool: ...


def _dummy_handler(ctx: CommandContext) -> CliResult[dict[str, bool]]:
    """Return a success result for testing.

    Parameters
    ----------
    ctx
        Handler context (unused in this test handler).

    Returns
    -------
    CliResult[dict[str, bool]]
        Success result with test data.
    """
    _ = ctx  # Acknowledge unused parameter
    return CliResult.ok({"test": True})


def _create_test_spec(
    operation_id: str = "test.op",
    name: str = "Test",
    description: str = "Test operation",
    group: str = "test",
    **kwargs: bool | tuple[str, ...],
) -> OperationSpec:
    """Create a test OperationSpec with configurable fields.

    Parameters
    ----------
    operation_id
        Operation identifier.
    name
        Operation name.
    description
        Operation description.
    group
        Operation group.
    **kwargs
        Additional spec fields: require_runtime, require_gateway,
        require_graph_runtime, tags, hidden.

    Returns
    -------
    OperationSpec
        Configured operation spec.
    """
    require_runtime = kwargs.get("require_runtime", True)
    require_gateway = kwargs.get("require_gateway", True)
    require_graph_runtime = kwargs.get("require_graph_runtime")
    tags_val = kwargs.get("tags", ())
    tags = tags_val if isinstance(tags_val, tuple) else ()
    hidden_val = kwargs.get("hidden")

    # Convert to proper boolean values
    require_runtime_bool = bool(require_runtime) if require_runtime is not None else True
    require_gateway_bool = bool(require_gateway) if require_gateway is not None else True
    require_graph_runtime_bool = bool(require_graph_runtime) if require_graph_runtime else False
    hidden_bool = bool(hidden_val) if hidden_val else False

    return OperationSpec(
        operation_id=operation_id,
        name=name,
        description=description,
        handler=_dummy_handler,
        group=group,
        require_runtime=require_runtime_bool,
        require_gateway=require_gateway_bool,
        require_graph_runtime=require_graph_runtime_bool,
        tags=tags,
        hidden=hidden_bool,
    )


def _verify_equal(actual: object, expected: object, message: str = "") -> None:
    """Verify two values are equal, raising on failure.

    Parameters
    ----------
    actual
        Actual value.
    expected
        Expected value.
    message
        Optional message for failure.

    Raises
    ------
    AssertionError
        If values are not equal.
    """
    if actual != expected:
        msg = f"Expected {expected!r}, got {actual!r}"
        if message:
            msg = f"{message}: {msg}"
        raise AssertionError(msg)


def _verify_true(*, condition: bool, message: str = "Expected True") -> None:
    """Verify a condition is true, raising on failure.

    Parameters
    ----------
    condition
        Condition to verify.
    message
        Message for failure.

    Raises
    ------
    AssertionError
        If condition is not true.
    """
    if not condition:
        raise AssertionError(message)


def _verify_false(*, condition: bool, message: str = "Expected False") -> None:
    """Verify a condition is false, raising on failure.

    Parameters
    ----------
    condition
        Condition to verify.
    message
        Message for failure.

    Raises
    ------
    AssertionError
        If condition is not false.
    """
    if condition:
        raise AssertionError(message)


def _verify_is(actual: object, expected: object, message: str = "") -> None:
    """Verify two objects are the same instance.

    Parameters
    ----------
    actual
        Actual object.
    expected
        Expected object.
    message
        Optional message for failure.

    Raises
    ------
    AssertionError
        If objects are not the same.
    """
    if actual is not expected:
        base_msg = "Expected same instance, got different objects"
        full_msg = f"{message}: {base_msg}" if message else base_msg
        raise AssertionError(full_msg)


def _verify_is_not(actual: object, expected: object, message: str = "") -> None:
    """Verify two objects are not the same instance.

    Parameters
    ----------
    actual
        Actual object.
    expected
        Expected object (should be different).
    message
        Optional message for failure.

    Raises
    ------
    AssertionError
        If objects are the same.
    """
    if actual is expected:
        base_msg = "Expected different instances"
        full_msg = f"{message}: {base_msg}" if message else base_msg
        raise AssertionError(full_msg)


def _verify_none(actual: object, message: str = "Expected None") -> None:
    """Verify a value is None.

    Parameters
    ----------
    actual
        Actual value.
    message
        Message for failure.

    Raises
    ------
    AssertionError
        If value is not None.
    """
    if actual is not None:
        full_msg = message + ": got " + repr(actual)
        raise AssertionError(full_msg)


def _verify_in(item: str, container: _StrContainer, message: str = "") -> None:
    """Verify an item is in a container.

    Parameters
    ----------
    item
        Item to check.
    container
        Container to check in.
    message
        Optional message for failure.

    Raises
    ------
    AssertionError
        If item is not in container.
    """
    if item not in container:
        msg = f"{item!r} not in container"
        if message:
            msg = f"{message}: {msg}"
        raise AssertionError(msg)


def _verify_not_in(item: str, container: _StrContainer, message: str = "") -> None:
    """Verify an item is not in a container.

    Parameters
    ----------
    item
        Item to check.
    container
        Container to check in.
    message
        Optional message for failure.

    Raises
    ------
    AssertionError
        If item is in container.
    """
    if item in container:
        msg = f"{item!r} unexpectedly in container"
        if message:
            msg = f"{message}: {msg}"
        raise AssertionError(msg)


@pytest.fixture(autouse=True)
def _reset_global_registry() -> None:
    """Reset global registry before each test."""
    reset_registry()


# -----------------------------------------------------------------------------
# OperationSpec Tests
# -----------------------------------------------------------------------------


def test_spec_creation_required_fields() -> None:
    """Create operation spec with required fields only."""
    spec = _create_test_spec(
        operation_id="test.op",
        name="Test Operation",
        description="A test operation",
        group="test",
    )
    _verify_equal(spec.operation_id, "test.op")
    _verify_equal(spec.name, "Test Operation")
    _verify_equal(spec.description, "A test operation")
    _verify_is(spec.handler, _dummy_handler)
    _verify_equal(spec.group, "test")


def test_spec_creation_with_defaults() -> None:
    """Create operation spec uses correct defaults."""
    spec = _create_test_spec()
    _verify_true(condition=spec.require_runtime, message="require_runtime should default True")
    _verify_true(condition=spec.require_gateway, message="require_gateway should default True")
    _verify_false(
        condition=spec.require_graph_runtime, message="require_graph_runtime should default False"
    )
    _verify_equal(spec.tags, ())
    _verify_false(condition=spec.hidden, message="hidden should default False")


def test_spec_creation_with_all_fields() -> None:
    """Create operation spec with all fields."""
    spec = _create_test_spec(
        operation_id="test.op",
        name="Test Operation",
        description="A test operation",
        group="test",
        require_runtime=False,
        require_gateway=False,
        require_graph_runtime=True,
        tags=("tag1", "tag2"),
        hidden=True,
    )
    _verify_false(condition=spec.require_runtime, message="require_runtime should be False")
    _verify_false(condition=spec.require_gateway, message="require_gateway should be False")
    _verify_true(
        condition=spec.require_graph_runtime, message="require_graph_runtime should be True"
    )
    _verify_equal(spec.tags, ("tag1", "tag2"))
    _verify_true(condition=spec.hidden, message="hidden should be True")


def test_spec_immutability() -> None:
    """Verify OperationSpec is frozen."""
    spec = _create_test_spec()
    with pytest.raises(AttributeError):
        spec.operation_id = "other.op"  # type: ignore[misc]


# -----------------------------------------------------------------------------
# OperationRegistry Tests
# -----------------------------------------------------------------------------


def test_registry_register_operation() -> None:
    """Register operation successfully."""
    registry = OperationRegistry()
    spec = _create_test_spec()
    result = registry.register(spec)
    _verify_is(result, spec)
    _verify_in("test.op", registry)
    _verify_equal(len(registry), EXPECTED_COUNT_ONE)


def test_registry_register_duplicate_raises() -> None:
    """Registering duplicate ID raises ValueError."""
    registry = OperationRegistry()
    spec = _create_test_spec()
    registry.register(spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_get_existing() -> None:
    """Get existing operation."""
    registry = OperationRegistry()
    spec = _create_test_spec()
    registry.register(spec)

    result = registry.get("test.op")
    _verify_is(result, spec)


def test_registry_get_missing_returns_none() -> None:
    """Get missing operation returns None."""
    registry = OperationRegistry()
    _verify_none(registry.get("nonexistent"))


def test_registry_require_existing() -> None:
    """Require existing operation."""
    registry = OperationRegistry()
    spec = _create_test_spec()
    registry.register(spec)

    result = registry.require("test.op")
    _verify_is(result, spec)


def test_registry_require_missing_raises() -> None:
    """Require missing operation raises KeyError."""
    registry = OperationRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.require("nonexistent")


def test_registry_list_operations_empty() -> None:
    """List operations from empty registry."""
    registry = OperationRegistry()
    ops = registry.list_operations()
    _verify_equal(ops, [])


def test_registry_list_operations_sorted() -> None:
    """List all operations sorted by ID."""
    registry = OperationRegistry()
    registry.register(_create_test_spec(operation_id="b.op", name="B", group="b"))
    registry.register(_create_test_spec(operation_id="a.op", name="A", group="a"))

    ops = registry.list_operations()
    _verify_equal(len(ops), EXPECTED_COUNT_TWO)
    _verify_equal(ops[0].operation_id, "a.op")
    _verify_equal(ops[1].operation_id, "b.op")


def test_registry_list_operations_by_group() -> None:
    """List operations filtered by group."""
    registry = OperationRegistry()
    registry.register(_create_test_spec(operation_id="jobs.list", name="List", group="jobs"))
    registry.register(_create_test_spec(operation_id="build.run", name="Run", group="build"))

    ops = registry.list_operations(group="jobs")
    _verify_equal(len(ops), EXPECTED_COUNT_ONE)
    _verify_equal(ops[0].operation_id, "jobs.list")


def test_registry_list_excludes_hidden() -> None:
    """List excludes hidden operations by default."""
    registry = OperationRegistry()
    registry.register(_create_test_spec(operation_id="visible", name="Visible"))
    registry.register(_create_test_spec(operation_id="hidden", name="Hidden", hidden=True))

    ops = registry.list_operations()
    _verify_equal(len(ops), EXPECTED_COUNT_ONE)
    _verify_equal(ops[0].operation_id, "visible")

    ops_with_hidden = registry.list_operations(include_hidden=True)
    _verify_equal(len(ops_with_hidden), EXPECTED_COUNT_TWO)


def test_registry_list_groups() -> None:
    """List all groups sorted."""
    registry = OperationRegistry()
    registry.register(_create_test_spec(operation_id="build.run", group="build"))
    registry.register(_create_test_spec(operation_id="jobs.list", group="jobs"))
    registry.register(_create_test_spec(operation_id="jobs.status", group="jobs"))

    groups = registry.list_groups()
    _verify_equal(groups, ["build", "jobs"])


def test_registry_unregister_existing() -> None:
    """Unregister existing operation."""
    registry = OperationRegistry()
    registry.register(_create_test_spec())
    _verify_in("test.op", registry)

    result = registry.unregister("test.op")
    _verify_true(condition=result, message="unregister should return True")
    _verify_not_in("test.op", registry)


def test_registry_unregister_missing() -> None:
    """Unregister missing operation returns False."""
    registry = OperationRegistry()
    result = registry.unregister("nonexistent")
    _verify_false(condition=result, message="unregister should return False for missing")


def test_registry_clear() -> None:
    """Clear all operations."""
    registry = OperationRegistry()
    registry.register(_create_test_spec())
    _verify_equal(len(registry), EXPECTED_COUNT_ONE)

    registry.clear()
    _verify_equal(len(registry), EXPECTED_COUNT_ZERO)


def test_registry_contains() -> None:
    """Check if operation is registered via 'in'."""
    registry = OperationRegistry()
    registry.register(_create_test_spec())
    _verify_in("test.op", registry)
    _verify_not_in("other.op", registry)


# -----------------------------------------------------------------------------
# Global Registry Tests
# -----------------------------------------------------------------------------


def test_global_get_registry_returns_same_instance() -> None:
    """get_registry returns same instance on repeated calls."""
    r1 = get_registry()
    r2 = get_registry()
    _verify_is(r1, r2)


def test_global_reset_creates_new_instance() -> None:
    """reset_registry causes new instance to be created."""
    r1 = get_registry()
    reset_registry()
    r2 = get_registry()
    _verify_is_not(r1, r2)


def test_global_register_operation_uses_global() -> None:
    """register_operation adds to global registry."""
    spec = _create_test_spec(operation_id="global.test")
    register_operation(spec)

    registry = get_registry()
    _verify_in("global.test", registry)
    _verify_is(registry.get("global.test"), spec)


def test_global_register_operation_returns_spec() -> None:
    """register_operation returns the registered spec."""
    spec = _create_test_spec(operation_id="global.test")
    result = register_operation(spec)
    _verify_is(result, spec)


# -----------------------------------------------------------------------------
# Group and Tag Filtering Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def populated_registry() -> OperationRegistry:
    """Create a registry with multiple operations for filtering tests.

    Returns
    -------
    OperationRegistry
        Registry populated with test operations across groups.
    """
    registry = OperationRegistry()

    registry.register(
        _create_test_spec(
            operation_id="jobs.list",
            name="List Jobs",
            description="List jobs",
            group="jobs",
            tags=("list", "read"),
        )
    )
    registry.register(
        _create_test_spec(
            operation_id="jobs.cancel",
            name="Cancel Job",
            description="Cancel job",
            group="jobs",
            tags=("write",),
        )
    )
    registry.register(
        _create_test_spec(
            operation_id="build.run",
            name="Build Run",
            description="Run build",
            group="build",
            tags=("write",),
        )
    )
    registry.register(
        _create_test_spec(
            operation_id="build.status",
            name="Build Status",
            description="Build status",
            group="build",
            tags=("read",),
        )
    )
    registry.register(
        _create_test_spec(
            operation_id="internal.debug",
            name="Debug",
            description="Debug",
            group="internal",
            hidden=True,
        )
    )

    return registry


def test_filter_by_group(populated_registry: OperationRegistry) -> None:
    """Filter operations by specific group."""
    jobs = populated_registry.list_operations(group="jobs")
    _verify_equal(len(jobs), EXPECTED_COUNT_TWO)
    for op in jobs:
        _verify_equal(op.group, "jobs")


def test_filter_by_nonexistent_group(populated_registry: OperationRegistry) -> None:
    """Filter by nonexistent group returns empty list."""
    ops = populated_registry.list_operations(group="nonexistent")
    _verify_equal(ops, [])


def test_include_hidden(populated_registry: OperationRegistry) -> None:
    """Include hidden operations when requested."""
    visible = populated_registry.list_operations()
    all_ops = populated_registry.list_operations(include_hidden=True)

    _verify_equal(len(visible), EXPECTED_COUNT_FOUR)
    _verify_equal(len(all_ops), EXPECTED_COUNT_FIVE)


def test_hidden_and_group_filter(populated_registry: OperationRegistry) -> None:
    """Combine hidden and group filters."""
    internal = populated_registry.list_operations(group="internal", include_hidden=True)
    _verify_equal(len(internal), EXPECTED_COUNT_ONE)
    _verify_equal(internal[0].operation_id, "internal.debug")

    visible_internal = populated_registry.list_operations(group="internal")
    _verify_equal(len(visible_internal), EXPECTED_COUNT_ZERO)
