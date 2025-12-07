"""Tests for the operation to targets mapping module.

These tests verify that operations correctly map their requirements
(datasets, graphs) to build system targets.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from codeintel.build.operations import (
    OperationTargets,
    get_all_operation_targets,
    get_targets_for_operation,
    resolve_targets_for_operation,
)
from codeintel.serving.operations.catalog import get_operation

# =============================================================================
# OperationTargets Dataclass Tests
# =============================================================================


def test_operation_targets_empty() -> None:
    """Verify empty OperationTargets is valid."""
    targets = OperationTargets(
        operation_id="test.op",
        required_targets=frozenset(),
        graph_targets=frozenset(),
        data_targets=frozenset(),
    )

    assert targets.operation_id == "test.op"
    assert len(targets.required_targets) == 0
    assert len(targets.graph_targets) == 0
    assert len(targets.data_targets) == 0


def test_operation_targets_with_data() -> None:
    """Verify OperationTargets stores targets correctly."""
    targets = OperationTargets(
        operation_id="test.op",
        required_targets=frozenset({"call_graph", "ast"}),
        graph_targets=frozenset({"call_graph"}),
        data_targets=frozenset({"ast"}),
    )

    assert targets.operation_id == "test.op"
    assert "call_graph" in targets.required_targets
    assert "ast" in targets.required_targets
    assert "call_graph" in targets.graph_targets
    assert "ast" in targets.data_targets


def test_operation_targets_frozen() -> None:
    """Verify OperationTargets is immutable."""
    targets = OperationTargets(
        operation_id="test.op",
        required_targets=frozenset({"call_graph"}),
        graph_targets=frozenset({"call_graph"}),
        data_targets=frozenset(),
    )

    with pytest.raises(FrozenInstanceError):
        cast("Any", targets).operation_id = "other"


# =============================================================================
# Target Resolution Tests
# =============================================================================


def test_get_targets_for_unknown_operation() -> None:
    """Verify unknown operation returns empty targets."""
    targets = get_targets_for_operation("nonexistent.operation")

    assert targets.operation_id == "nonexistent.operation"
    assert len(targets.required_targets) == 0


def test_get_targets_for_function_summary() -> None:
    """Verify function.summary maps to call_graph target."""
    targets = get_targets_for_operation("function.summary")

    assert targets.operation_id == "function.summary"
    assert "call_graph" in targets.graph_targets
    assert "call_graph" in targets.required_targets


def test_get_targets_for_dataset_list() -> None:
    """Verify datasets.list has no required targets."""
    targets = get_targets_for_operation("datasets.list")

    assert targets.operation_id == "datasets.list"
    # datasets.list has no required_datasets or required_graphs
    assert len(targets.required_targets) == 0


def test_get_targets_for_graph_call_neighborhood() -> None:
    """Verify graph.call_neighborhood maps to call_graph target."""
    targets = get_targets_for_operation("graph.call_neighborhood")

    assert targets.operation_id == "graph.call_neighborhood"
    assert "call_graph" in targets.graph_targets
    # Also has call_graph_nodes in required_datasets
    assert "call_graph" in targets.data_targets or "call_graph" in targets.graph_targets


def test_get_targets_for_import_boundary() -> None:
    """Verify graph.import_boundary maps to import_graph target."""
    targets = get_targets_for_operation("graph.import_boundary")

    assert targets.operation_id == "graph.import_boundary"
    assert "import_graph" in targets.graph_targets


def test_get_targets_caching() -> None:
    """Verify get_targets_for_operation is cached."""
    targets1 = get_targets_for_operation("function.summary")
    targets2 = get_targets_for_operation("function.summary")

    assert targets1 is targets2  # Same object due to lru_cache


# =============================================================================
# Graph to Target Mapping Tests
# =============================================================================


def test_callgraph_maps_to_call_graph() -> None:
    """Verify 'callgraph' graph runtime maps to 'call_graph' target."""
    targets = get_targets_for_operation("function.summary")
    assert "call_graph" in targets.graph_targets


def test_importgraph_maps_to_import_graph() -> None:
    """Verify 'importgraph' graph runtime maps to 'import_graph' target."""
    targets = get_targets_for_operation("graph.import_boundary")
    assert "import_graph" in targets.graph_targets


# =============================================================================
# resolve_targets_for_operation Tests
# =============================================================================


def test_resolve_targets_for_operation_with_graphs() -> None:
    """Verify resolve_targets_for_operation handles graph requirements."""
    op = get_operation("function.summary")
    assert op is not None

    targets = resolve_targets_for_operation(op)

    assert targets.operation_id == "function.summary"
    assert "call_graph" in targets.graph_targets


def test_resolve_targets_for_operation_with_datasets() -> None:
    """Verify resolve_targets_for_operation handles dataset requirements."""
    op = get_operation("graph.call_neighborhood")
    assert op is not None

    targets = resolve_targets_for_operation(op)

    assert targets.operation_id == "graph.call_neighborhood"
    # call_graph_nodes is in required_datasets
    assert len(targets.data_targets) >= 1 or len(targets.graph_targets) >= 1


# =============================================================================
# get_all_operation_targets Tests
# =============================================================================


def test_get_all_operation_targets_returns_dict() -> None:
    """Verify get_all_operation_targets returns all operations."""
    all_targets = get_all_operation_targets()

    assert isinstance(all_targets, dict)
    assert len(all_targets) > 0


def test_get_all_operation_targets_includes_known_operations() -> None:
    """Verify all_operation_targets includes known operations."""
    all_targets = get_all_operation_targets()

    assert "function.summary" in all_targets
    assert "datasets.list" in all_targets
    assert "graph.call_neighborhood" in all_targets


def test_get_all_operation_targets_values_are_operation_targets() -> None:
    """Verify all values are OperationTargets instances."""
    all_targets = get_all_operation_targets()

    for op_id, targets in all_targets.items():
        assert isinstance(targets, OperationTargets)
        assert targets.operation_id == op_id


# =============================================================================
# Table to Target Mapping Tests
# =============================================================================


def test_table_to_target_mapping_exists() -> None:
    """Verify table to target mapping is built correctly."""
    # Force index building by resolving an operation
    targets = get_targets_for_operation("graph.call_neighborhood")

    # call_graph_nodes table should map to call_graph target
    # This is verified by checking graph.call_neighborhood has data_targets
    assert "call_graph" in targets.required_targets


def test_required_targets_is_union_of_graph_and_data() -> None:
    """Verify required_targets is union of graph_targets and data_targets."""
    targets = get_targets_for_operation("graph.callgraph.edges")

    expected_union = targets.graph_targets | targets.data_targets
    assert targets.required_targets == expected_union


# =============================================================================
# Edge Cases
# =============================================================================


def test_operation_with_multiple_graph_requirements() -> None:
    """Verify operation with multiple graph requirements maps correctly."""
    targets = get_targets_for_operation("architecture.function")

    # architecture.function requires both callgraph and importgraph
    assert "call_graph" in targets.graph_targets
    assert "import_graph" in targets.graph_targets


def test_operation_id_preserved_on_unknown() -> None:
    """Verify operation ID is preserved even for unknown operations."""
    op_id = "completely.unknown.operation.id"
    targets = get_targets_for_operation(op_id)

    assert targets.operation_id == op_id
