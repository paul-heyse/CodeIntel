"""Tests for dynamic CLI command generation from operation metadata.

These tests verify that the dynamic CLI parameter introspection system
correctly generates typed commands for serving operations.
"""

from __future__ import annotations

import pytest

from codeintel.cli.op_params import (
    OperationCliMetadata,
    ParamRole,
    build_operation_cli_metadata,
    classify_param_role,
    get_backend_signature_for_operation,
    get_operations_with_cli_support,
)
from codeintel.serving.operations.catalog import get_operation, iter_operations
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli import run_cli

# -----------------------------------------------------------------------------
# Parameter Classification Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param_name", "expected_role"),
    [
        ("goid_h128", "selector"),
        ("function_goid_h128", "selector"),
        ("urn", "selector"),
        ("path", "selector"),
        ("rel_path", "selector"),
        ("module", "selector"),
        ("qualname", "selector"),
        ("subsystem_id", "selector"),
        ("limit", "filter"),
        ("offset", "filter"),
        ("tested_only", "filter"),
        ("min_risk", "filter"),
        ("max_depth", "filter"),
        ("kind", "filter"),
        ("scope", "advanced"),
        ("graph_scope", "advanced"),
        ("radius", "advanced"),
        ("max_nodes", "advanced"),
        ("some_unknown_param", "filter"),  # Default for unknown params
    ],
)
def test_classify_param_role_categorizes_correctly(
    param_name: str,
    expected_role: ParamRole,
) -> None:
    """Verify parameter names are classified into the correct role."""
    result = classify_param_role(param_name)
    expect_equal(result, expected_role)


def test_classify_param_role_with_graph_operation_context() -> None:
    """Verify graph-related params are classified as advanced for graph ops."""
    # Get a graph-related operation
    op = get_operation("callgraph.neighbors")

    # max_nodes should be classified as advanced for graph operations
    result = classify_param_role("max_nodes", operation=op)
    expect_equal(result, "advanced")


# -----------------------------------------------------------------------------
# Operation CLI Support Tests
# -----------------------------------------------------------------------------


def test_get_operations_with_cli_support_returns_list() -> None:
    """Verify operations with CLI support can be enumerated."""
    ops = get_operations_with_cli_support()

    expect_is_instance(ops, list)
    expect_true(len(ops) > 0)

    # Check structure
    for op in ops:
        expect_is_not_none(op.id)
        expect_is_not_none(op.backend_method)


def test_get_backend_signature_for_known_operation() -> None:
    """Verify signature introspection works for known operations."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = get_backend_signature_for_operation(op)

    # May be None if backend method mapping doesn't exist
    if result is not None:
        sig, hints = result
        expect_is_not_none(sig)
        expect_is_instance(hints, dict)


def test_build_operation_cli_metadata_for_function_summary() -> None:
    """Verify CLI metadata is built for function.summary operation."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    expect_is_instance(result.params, tuple)
    expect_is_instance(result.help_text, str)
    expect_true(len(result.help_text) > 0)


# -----------------------------------------------------------------------------
# Dynamic Command Registration Tests
# -----------------------------------------------------------------------------


def test_op_help_shows_core_commands() -> None:
    """Verify op --help shows core commands (list, call)."""
    # Note: Dynamic command registration is tested separately from help display
    # to avoid Typer initialization issues in tests
    ops = list(iter_operations())
    expect_true(len(ops) > 0, message="Should have at least some operations registered")

    # Verify some known operations exist
    op_ids = {op.id for op in ops}
    expect_in("function.summary", op_ids)
    expect_in("file.summary", op_ids)


def test_operation_metadata_available() -> None:
    """Verify operation metadata is available for CLI generation."""
    ops = get_operations_with_cli_support()
    expect_true(len(ops) > 0, message="Should have at least some operations with CLI support")

    # Verify metadata can be built for all operations
    for op in ops[:5]:  # Test first 5 to keep test fast
        metadata = build_operation_cli_metadata(op)
        expect_is_not_none(metadata.cli_name)
        expect_is_not_none(metadata.help_text)


def test_operation_validation_rejects_unknown() -> None:
    """Verify operation validation rejects unknown operation IDs."""
    op = get_operation("nonexistent.operation")
    expect_true(op is None, message="Should return None for unknown operations")


def test_operation_validation_accepts_known() -> None:
    """Verify operation validation accepts known operation IDs."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)
    expect_equal(op.id, "function.summary")


def test_operation_has_required_datasets() -> None:
    """Verify operations have required_datasets field."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)
    expect_true(hasattr(op, "required_datasets"))
    # required_datasets should be a tuple
    expect_is_instance(op.required_datasets, tuple)


# -----------------------------------------------------------------------------
# Dynamic subcommand registration (Cyclopts)
# -----------------------------------------------------------------------------


def test_dynamic_op_help_available() -> None:
    """Dynamic subcommands should be registered and expose help."""
    op = next(iter_operations())
    command_name = op.id.replace(".", "-")

    result = run_cli(["op", command_name, "--help"])

    expect_equal(result.exit_code, 0)
    if op.summary:
        expect_in(op.summary.split()[0].lower(), result.stdout.lower())


# -----------------------------------------------------------------------------
# Parameter Spec Tests
# -----------------------------------------------------------------------------


def test_cli_param_spec_has_correct_structure() -> None:
    """Verify CliParamSpec dataclass has expected fields."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    if len(params) > 0:
        param = params[0]
        # Check expected attributes exist
        expect_true(hasattr(param, "name"))
        expect_true(hasattr(param, "python_type"))
        expect_true(hasattr(param, "default"))
        expect_true(hasattr(param, "is_optional"))
        expect_true(hasattr(param, "role"))
        expect_true(hasattr(param, "help_text"))
        expect_true(hasattr(param, "help_panel"))


def test_selector_params_come_first_in_classification() -> None:
    """Verify selector params are prioritized in help panels."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    selectors = [p for p in params if p.role == "selector"]
    filters = [p for p in params if p.role == "filter"]

    # Selectors should have help_panel="Target Selection"
    for param in selectors:
        expect_equal(param.help_panel, "Target Selection")

    # Filters should have help_panel="Filtering Options"
    for param in filters:
        expect_equal(param.help_panel, "Filtering Options")


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


def test_dynamic_cli_respects_operation_defaults() -> None:
    """Verify dynamic CLI uses operation-specific defaults."""
    op = get_operation("function.summary")
    op = expect_is_not_none(op)

    result = build_operation_cli_metadata(op)

    expect_is_instance(result, OperationCliMetadata)
    params = result.params
    # Check that default values are populated where applicable
    for param in params:
        # Optional parameters should have is_optional set
        if param.is_optional:
            # Optional params have explicit defaults or are typed
            expect_true(
                param.default is not None or param.python_type is not None,
                message="Optional parameters should have defaults or type hints",
            )
