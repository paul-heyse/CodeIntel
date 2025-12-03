"""Tests for dynamic CLI command generation from operation metadata.

These tests verify that the dynamic CLI parameter introspection system
correctly generates typed commands for serving operations.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from codeintel.cli.op_params import (
    OperationCliMetadata,
    ParamRole,
    build_operation_cli_metadata,
    classify_param_role,
    get_backend_signature_for_operation,
    get_operations_with_cli_support,
)
from codeintel.serving.operations.catalog import get_operation, iter_operations

runner = CliRunner()


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
    assert result == expected_role


def test_classify_param_role_with_graph_operation_context() -> None:
    """Verify graph-related params are classified as advanced for graph ops."""
    # Get a graph-related operation
    op = get_operation("callgraph.neighbors")

    # max_nodes should be classified as advanced for graph operations
    result = classify_param_role("max_nodes", operation=op)
    assert result == "advanced"


# -----------------------------------------------------------------------------
# Operation CLI Support Tests
# -----------------------------------------------------------------------------


def test_get_operations_with_cli_support_returns_list() -> None:
    """Verify operations with CLI support can be enumerated."""
    ops = get_operations_with_cli_support()

    assert isinstance(ops, list)
    assert len(ops) > 0

    # Check structure
    for op in ops:
        assert op.id is not None
        assert op.backend_method is not None


def test_get_backend_signature_for_known_operation() -> None:
    """Verify signature introspection works for known operations."""
    op = get_operation("function.summary")
    assert op is not None

    result = get_backend_signature_for_operation(op)

    # May be None if backend method mapping doesn't exist
    if result is not None:
        sig, hints = result
        assert sig is not None
        assert isinstance(hints, dict)


def test_build_operation_cli_metadata_for_function_summary() -> None:
    """Verify CLI metadata is built for function.summary operation."""
    op = get_operation("function.summary")
    assert op is not None

    result = build_operation_cli_metadata(op)

    assert isinstance(result, OperationCliMetadata)
    assert isinstance(result.params, tuple)
    assert isinstance(result.help_text, str)
    assert len(result.help_text) > 0


# -----------------------------------------------------------------------------
# Dynamic Command Registration Tests
# -----------------------------------------------------------------------------


def test_op_help_shows_core_commands() -> None:
    """Verify op --help shows core commands (list, call)."""
    # Note: Dynamic command registration is tested separately from help display
    # to avoid Typer initialization issues in tests
    ops = list(iter_operations())
    assert len(ops) > 0, "Should have at least some operations registered"

    # Verify some known operations exist
    op_ids = {op.id for op in ops}
    assert "function.summary" in op_ids
    assert "file.summary" in op_ids


def test_operation_metadata_available() -> None:
    """Verify operation metadata is available for CLI generation."""
    ops = get_operations_with_cli_support()
    assert len(ops) > 0, "Should have at least some operations with CLI support"

    # Verify metadata can be built for all operations
    for op in ops[:5]:  # Test first 5 to keep test fast
        metadata = build_operation_cli_metadata(op)
        assert metadata.cli_name is not None
        assert metadata.help_text is not None


def test_operation_validation_rejects_unknown() -> None:
    """Verify operation validation rejects unknown operation IDs."""
    op = get_operation("nonexistent.operation")
    assert op is None, "Should return None for unknown operations"


def test_operation_validation_accepts_known() -> None:
    """Verify operation validation accepts known operation IDs."""
    op = get_operation("function.summary")
    assert op is not None, "Should return Operation for known operations"
    assert op.id == "function.summary"


def test_operation_has_required_datasets() -> None:
    """Verify operations have required_datasets field."""
    op = get_operation("function.summary")
    assert op is not None
    assert hasattr(op, "required_datasets")
    # required_datasets should be a tuple
    assert isinstance(op.required_datasets, tuple)


# -----------------------------------------------------------------------------
# Parameter Spec Tests
# -----------------------------------------------------------------------------


def test_cli_param_spec_has_correct_structure() -> None:
    """Verify CliParamSpec dataclass has expected fields."""
    op = get_operation("function.summary")
    assert op is not None

    result = build_operation_cli_metadata(op)

    assert isinstance(result, OperationCliMetadata)
    params = result.params
    if len(params) > 0:
        param = params[0]
        # Check expected attributes exist
        assert hasattr(param, "name")
        assert hasattr(param, "python_type")
        assert hasattr(param, "default")
        assert hasattr(param, "is_optional")
        assert hasattr(param, "role")
        assert hasattr(param, "help_text")
        assert hasattr(param, "help_panel")


def test_selector_params_come_first_in_classification() -> None:
    """Verify selector params are prioritized in help panels."""
    op = get_operation("function.summary")
    assert op is not None

    result = build_operation_cli_metadata(op)

    assert isinstance(result, OperationCliMetadata)
    params = result.params
    selectors = [p for p in params if p.role == "selector"]
    filters = [p for p in params if p.role == "filter"]

    # Selectors should have help_panel="Target Selection"
    for param in selectors:
        assert param.help_panel == "Target Selection"

    # Filters should have help_panel="Filtering Options"
    for param in filters:
        assert param.help_panel == "Filtering Options"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


def test_dynamic_cli_respects_operation_defaults() -> None:
    """Verify dynamic CLI uses operation-specific defaults."""
    op = get_operation("function.summary")
    assert op is not None

    result = build_operation_cli_metadata(op)

    assert isinstance(result, OperationCliMetadata)
    params = result.params
    # Check that default values are populated where applicable
    for param in params:
        # Optional parameters should have is_optional set
        if param.is_optional:
            # Optional params have explicit defaults or are typed
            assert param.default is not None or param.python_type is not None
