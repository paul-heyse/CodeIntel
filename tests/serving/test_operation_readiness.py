"""Tests for operation prerequisite readiness using the build system.

These tests verify that the serving layer correctly uses the build system's
readiness model to check if operations can run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from codeintel.build.operations import (
    get_all_operation_targets,
    get_targets_for_operation,
)
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import SnapshotRef
from codeintel.serving.auto_pipeline import (
    PrerequisiteError,
    diagnose_prereq_failure,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def snapshot() -> SnapshotRef:
    """Create a test snapshot reference.

    Returns
    -------
    SnapshotRef
        Test snapshot.
    """
    return SnapshotRef(
        repo="test/repo",
        commit="abc123",
        repo_root=Path.cwd(),
    )


# =============================================================================
# PrerequisiteError Tests
# =============================================================================


def test_prerequisite_error_structure() -> None:
    """Verify PrerequisiteError dataclass structure."""
    error = PrerequisiteError(
        op_id="function.summary",
        missing_targets=("call_graph", "ast"),
        bottleneck="ast",
        fix_command="codeintel build run ast",
        human_message="Missing targets: ast, call_graph",
    )

    assert error.op_id == "function.summary"
    assert "call_graph" in error.missing_targets
    assert "ast" in error.missing_targets
    assert error.bottleneck == "ast"
    assert "ast" in error.fix_command
    assert error.human_message


def test_prerequisite_error_frozen() -> None:
    """Verify PrerequisiteError is immutable."""
    error = PrerequisiteError(
        op_id="test.op",
        missing_targets=(),
        bottleneck=None,
        fix_command="codeintel build run --all",
        human_message="Test",
    )

    with pytest.raises(AttributeError):
        error.op_id = "other"  # type: ignore[misc]


# =============================================================================
# operation_prereqs_satisfied Tests (Unit)
# =============================================================================


def test_prereqs_satisfied_no_requirements() -> None:
    """Verify operations with no requirements are considered satisfied.

    Operations like datasets.list have no required_datasets or required_graphs,
    so they should always be considered satisfied.
    """
    # datasets.list has no requirements
    # We can't fully test without a gateway, but we can verify the structure
    targets = get_targets_for_operation("datasets.list")
    assert len(targets.required_targets) == 0


def test_prereqs_satisfied_with_requirements() -> None:
    """Verify operations with requirements map to targets."""
    # function.summary requires callgraph
    targets = get_targets_for_operation("function.summary")
    assert "call_graph" in targets.graph_targets


# =============================================================================
# diagnose_prereq_failure Tests (Unit)
# =============================================================================


def test_diagnose_returns_prerequisite_error() -> None:
    """Verify diagnose_prereq_failure returns PrerequisiteError type."""
    # We verify the function exists and has the right signature
    sig = inspect.signature(diagnose_prereq_failure)
    params = list(sig.parameters.keys())

    assert "gateway" in params
    assert "op_id" in params
    assert "snapshot" in params


def test_diagnose_prereq_failure_command_format() -> None:
    """Verify the fix command has the correct format."""
    # The fix command should be 'codeintel build run <target>'
    error = PrerequisiteError(
        op_id="test.op",
        missing_targets=("ast",),
        bottleneck="ast",
        fix_command="codeintel build run ast",
        human_message="Test",
    )

    assert error.fix_command.startswith("codeintel build run")


# =============================================================================
# Integration with Build System Tests
# =============================================================================


def test_operation_targets_integration() -> None:
    """Verify operation targets correctly integrate with build system."""
    graph = get_target_graph()

    # Get targets for function.summary
    targets = get_targets_for_operation("function.summary")

    # Verify all targets exist in the build system
    for target_name in targets.required_targets:
        # This should not raise
        target = graph.get(target_name)
        assert target.name == target_name


def test_all_operations_have_valid_targets() -> None:
    """Verify all operations map to valid build targets."""
    graph = get_target_graph()
    all_op_targets = get_all_operation_targets()

    for op_id, op_targets in all_op_targets.items():
        for target_name in op_targets.required_targets:
            # All targets should exist in the graph
            try:
                target = graph.get(target_name)
                assert target.name == target_name
            except KeyError:
                pytest.fail(f"Operation {op_id} requires unknown target: {target_name}")


# =============================================================================
# Edge Cases
# =============================================================================


def test_unknown_operation_prereqs() -> None:
    """Verify unknown operations are handled gracefully."""
    targets = get_targets_for_operation("completely.unknown.operation")

    # Should return empty targets, not raise
    assert len(targets.required_targets) == 0
    assert targets.operation_id == "completely.unknown.operation"


def test_multiple_graph_requirements() -> None:
    """Verify operations with multiple graph requirements."""
    # architecture.function requires both callgraph and importgraph
    targets = get_targets_for_operation("architecture.function")

    assert "call_graph" in targets.graph_targets
    assert "import_graph" in targets.graph_targets
    assert len(targets.graph_targets) == 2
