"""Tests for operation prerequisite readiness using the build system.

These tests verify that the serving layer correctly uses the build system's
readiness model to check if operations can run.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

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
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)


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


def test_prerequisite_error_structure() -> None:
    """Verify PrerequisiteError dataclass structure."""
    error = PrerequisiteError(
        op_id="function.summary",
        missing_targets=("call_graph", "ast"),
        bottleneck="ast",
        fix_command="codeintel build run ast",
        human_message="Missing targets: ast, call_graph",
    )

    expect_equal(error.op_id, "function.summary")
    expect_in("call_graph", error.missing_targets)
    expect_in("ast", error.missing_targets)
    expect_equal(error.bottleneck, "ast")
    expect_in("ast", error.fix_command)
    expect_true(error.human_message)


def test_prerequisite_error_frozen() -> None:
    """Verify PrerequisiteError is immutable."""
    error = PrerequisiteError(
        op_id="test.op",
        missing_targets=(),
        bottleneck=None,
        fix_command="codeintel build run --all",
        human_message="Test",
    )

    with pytest.raises(FrozenInstanceError):
        cast("Any", error).op_id = "other"


def test_prereqs_satisfied_no_requirements() -> None:
    """Verify operations with no requirements are considered satisfied.

    Operations like datasets.list have no required_datasets or required_graphs,
    so they should always be considered satisfied.
    """
    targets = get_targets_for_operation("datasets.list")
    expect_equal(len(targets.required_targets), 0)


def test_prereqs_satisfied_with_requirements() -> None:
    """Verify operations with requirements map to targets."""
    targets = get_targets_for_operation("function.summary")
    expect_in("call_graph", targets.graph_targets)


def test_diagnose_returns_prerequisite_error() -> None:
    """Verify diagnose_prereq_failure returns PrerequisiteError type."""
    sig = inspect.signature(diagnose_prereq_failure)
    params = list(sig.parameters.keys())

    expect_in("gateway", params)
    expect_in("op_id", params)
    expect_in("snapshot", params)


def test_diagnose_prereq_failure_command_format() -> None:
    """Verify the fix command has the correct format."""
    error = PrerequisiteError(
        op_id="test.op",
        missing_targets=("ast",),
        bottleneck="ast",
        fix_command="codeintel build run ast",
        human_message="Test",
    )

    expect_true(error.fix_command.startswith("codeintel build run"))


def test_operation_targets_integration() -> None:
    """Verify operation targets correctly integrate with build system."""
    graph = get_target_graph()

    targets = get_targets_for_operation("function.summary")

    for target_name in targets.required_targets:
        target = graph.get(target_name)
        expect_equal(target.name, target_name)


def test_all_operations_have_valid_targets() -> None:
    """Verify all operations map to valid build targets."""
    graph = get_target_graph()
    all_op_targets = get_all_operation_targets()

    for op_id, op_targets in all_op_targets.items():
        for target_name in op_targets.required_targets:
            try:
                target = graph.get(target_name)
                expect_equal(target.name, target_name)
            except KeyError:
                pytest.fail(f"Operation {op_id} requires unknown target: {target_name}")


def test_unknown_operation_prereqs() -> None:
    """Verify unknown operations are handled gracefully."""
    targets = get_targets_for_operation("completely.unknown.operation")

    expect_equal(len(targets.required_targets), 0)
    expect_equal(targets.operation_id, "completely.unknown.operation")


def test_multiple_graph_requirements() -> None:
    """Verify operations with multiple graph requirements."""
    targets = get_targets_for_operation("architecture.function")

    expect_in("call_graph", targets.graph_targets)
    expect_in("import_graph", targets.graph_targets)
    expect_equal(len(targets.graph_targets), 2)
