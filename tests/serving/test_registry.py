"""Tests for serving layer registry module.

This module tests the dataset registry and dataflow graph building functionality
using real DATASET_CONTRACTS and operations from the canonical catalogs.
"""

from __future__ import annotations

import dataclasses

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.registry import (
    DatasetMeta,
    build_serving_dataflow_graph,
    get_registry_operation,
    iter_graph_nodes,
    iter_operation_dataset_edges,
    iter_operation_graph_edges,
    iter_operation_nodes,
    iter_registry_operations,
)

# Constants for test values
TUPLE_LENGTH = 2
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
SMALL_DEFAULT_LIMIT = 50
SMALL_MAX_LIMIT = 500

# =============================================================================
# iter_registry_operations Tests
# =============================================================================


def test_iter_registry_operations_returns_list() -> None:
    """Verify iter_registry_operations returns a non-empty list."""
    operations = iter_registry_operations()

    assert isinstance(operations, list)
    assert len(operations) > 0


def test_iter_registry_operations_all_have_ids() -> None:
    """Verify all operations have non-empty IDs."""
    operations = iter_registry_operations()

    for op in operations:
        assert hasattr(op, "id")
        assert op.id
        assert isinstance(op.id, str)


def test_iter_registry_operations_includes_datasets_rows() -> None:
    """Verify datasets.rows operation exists and has exposed_datasets patched."""
    operations = iter_registry_operations()
    datasets_rows = next((op for op in operations if op.id == "datasets.rows"), None)

    assert datasets_rows is not None
    assert hasattr(datasets_rows, "exposed_datasets")
    # Should be patched with DATASET_CONTRACTS_BY_TABLE_KEY keys
    assert len(datasets_rows.exposed_datasets) > 0


def test_iter_registry_operations_patched_datasets_match_contracts() -> None:
    """Verify datasets.rows exposed_datasets match actual contract keys."""
    operations = iter_registry_operations()
    datasets_rows = next((op for op in operations if op.id == "datasets.rows"), None)

    assert datasets_rows is not None
    exposed_set = set(datasets_rows.exposed_datasets)
    contract_keys = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    # Exposed datasets should be exactly the contract keys
    assert exposed_set == contract_keys


# =============================================================================
# get_registry_operation Tests
# =============================================================================


def test_get_registry_operation_found() -> None:
    """Verify get_registry_operation returns operation by ID."""
    operations = iter_registry_operations()
    # Pick first operation ID
    first_op = operations[0]

    result = get_registry_operation(first_op.id)

    assert result is not None
    assert result.id == first_op.id


def test_get_registry_operation_not_found() -> None:
    """Verify get_registry_operation returns None for unknown ID."""
    result = get_registry_operation("non_existent_operation_xyz")

    assert result is None


def test_get_registry_operation_datasets_rows() -> None:
    """Verify datasets.rows can be retrieved."""
    result = get_registry_operation("datasets.rows")

    assert result is not None
    assert result.id == "datasets.rows"
    assert hasattr(result, "exposed_datasets")


# =============================================================================
# iter_operation_nodes Tests
# =============================================================================


def test_iter_operation_nodes_returns_list() -> None:
    """Verify iter_operation_nodes returns DataflowNode list."""
    nodes = iter_operation_nodes()

    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_iter_operation_nodes_all_dataflow_nodes() -> None:
    """Verify all returned items are DataflowNode instances."""
    nodes = iter_operation_nodes()

    for node in nodes:
        assert isinstance(node, DataflowNode)


def test_iter_operation_nodes_have_operation_kind() -> None:
    """Verify all operation nodes have kind='operation'."""
    nodes = iter_operation_nodes()

    for node in nodes:
        assert node.kind == "operation"


def test_iter_operation_nodes_have_serving_family() -> None:
    """Verify all operation nodes have family='serving'."""
    nodes = iter_operation_nodes()

    for node in nodes:
        assert node.family == "serving"


def test_iter_operation_nodes_match_operations() -> None:
    """Verify operation nodes match registered operations."""
    operations = iter_registry_operations()
    nodes = iter_operation_nodes()

    op_ids = {op.id for op in operations}
    node_ids = {node.id for node in nodes}

    assert op_ids == node_ids


# =============================================================================
# iter_graph_nodes Tests
# =============================================================================


def test_iter_graph_nodes_returns_list() -> None:
    """Verify iter_graph_nodes returns DataflowNode list."""
    nodes = iter_graph_nodes()

    assert isinstance(nodes, list)
    # May be empty if no operations require graphs


def test_iter_graph_nodes_all_dataflow_nodes() -> None:
    """Verify all returned items are DataflowNode instances."""
    nodes = iter_graph_nodes()

    for node in nodes:
        assert isinstance(node, DataflowNode)


def test_iter_graph_nodes_have_graph_kind() -> None:
    """Verify all graph nodes have kind='graph'."""
    nodes = iter_graph_nodes()

    for node in nodes:
        assert node.kind == "graph"


def test_iter_graph_nodes_have_graph_prefix_ids() -> None:
    """Verify all graph node IDs start with 'graph.'."""
    nodes = iter_graph_nodes()

    for node in nodes:
        assert node.id.startswith("graph.")


def test_iter_graph_nodes_derive_from_operations() -> None:
    """Verify graph nodes come from operation required_graphs."""
    operations = iter_registry_operations()
    nodes = iter_graph_nodes()

    # Collect all required graph names from operations
    required_graphs: set[str] = set()
    for op in operations:
        for graph_name in op.required_graphs:
            required_graphs.add(f"graph.{graph_name}")

    node_ids = {node.id for node in nodes}

    assert node_ids == required_graphs


# =============================================================================
# iter_operation_dataset_edges Tests
# =============================================================================


def test_iter_operation_dataset_edges_returns_list() -> None:
    """Verify iter_operation_dataset_edges returns DataflowEdge list."""
    edges = iter_operation_dataset_edges()

    assert isinstance(edges, list)


def test_iter_operation_dataset_edges_all_dataflow_edges() -> None:
    """Verify all returned items are DataflowEdge instances."""
    edges = iter_operation_dataset_edges()

    for edge in edges:
        assert isinstance(edge, DataflowEdge)


def test_iter_operation_dataset_edges_have_valid_types() -> None:
    """Verify all edges have 'reads' or 'exposes' edge_type."""
    edges = iter_operation_dataset_edges()

    valid_types = {"reads", "exposes"}
    for edge in edges:
        assert edge.edge_type in valid_types


def test_iter_operation_dataset_edges_dst_are_operations() -> None:
    """Verify all edge destinations are valid operation IDs."""
    operations = iter_registry_operations()
    edges = iter_operation_dataset_edges()

    op_ids = {op.id for op in operations}
    for edge in edges:
        assert edge.dst in op_ids


# =============================================================================
# iter_operation_graph_edges Tests
# =============================================================================


def test_iter_operation_graph_edges_returns_list() -> None:
    """Verify iter_operation_graph_edges returns DataflowEdge list."""
    edges = iter_operation_graph_edges()

    assert isinstance(edges, list)


def test_iter_operation_graph_edges_all_dataflow_edges() -> None:
    """Verify all returned items are DataflowEdge instances."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        assert isinstance(edge, DataflowEdge)


def test_iter_operation_graph_edges_have_depends_on_type() -> None:
    """Verify all graph edges have 'depends_on' edge_type."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        assert edge.edge_type == "depends_on"


def test_iter_operation_graph_edges_src_are_graph_nodes() -> None:
    """Verify all edge sources are graph node IDs."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        assert edge.src.startswith("graph.")


def test_iter_operation_graph_edges_dst_are_operations() -> None:
    """Verify all edge destinations are valid operation IDs."""
    operations = iter_registry_operations()
    edges = iter_operation_graph_edges()

    op_ids = {op.id for op in operations}
    for edge in edges:
        assert edge.dst in op_ids


# =============================================================================
# build_serving_dataflow_graph Tests
# =============================================================================


def test_build_serving_dataflow_graph_returns_tuple() -> None:
    """Verify build_serving_dataflow_graph returns nodes and edges tuple."""
    result = build_serving_dataflow_graph()

    assert isinstance(result, tuple)
    assert len(result) == TUPLE_LENGTH


def test_build_serving_dataflow_graph_nodes_list() -> None:
    """Verify first element is list of DataflowNode."""
    nodes, _ = build_serving_dataflow_graph()

    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, DataflowNode)


def test_build_serving_dataflow_graph_edges_list() -> None:
    """Verify second element is list of DataflowEdge."""
    _, edges = build_serving_dataflow_graph()

    assert isinstance(edges, list)
    for edge in edges:
        assert isinstance(edge, DataflowEdge)


def test_build_serving_dataflow_graph_includes_operations() -> None:
    """Verify graph includes operation nodes."""
    nodes, _ = build_serving_dataflow_graph()

    operation_nodes = [n for n in nodes if n.kind == "operation"]
    assert len(operation_nodes) > 0


def test_build_serving_dataflow_graph_includes_datasets() -> None:
    """Verify graph includes dataset nodes."""
    nodes, _ = build_serving_dataflow_graph()

    # Dataset nodes have kind 'dataset' or 'view'
    dataset_view_kinds = {"dataset", "view"}
    dataset_nodes = [n for n in nodes if n.kind in dataset_view_kinds]
    assert len(dataset_nodes) > 0


def test_build_serving_dataflow_graph_edges_deduplicated() -> None:
    """Verify edges are deduplicated."""
    _, edges = build_serving_dataflow_graph()

    edge_keys = [(e.src, e.dst, e.edge_type) for e in edges]
    unique_keys = set(edge_keys)

    assert len(edge_keys) == len(unique_keys)


def test_build_serving_dataflow_graph_nodes_deduplicated() -> None:
    """Verify nodes are deduplicated by (id, kind)."""
    nodes, _ = build_serving_dataflow_graph()

    node_keys = [(n.id, n.kind) for n in nodes]
    unique_keys = set(node_keys)

    assert len(node_keys) == len(unique_keys)


# =============================================================================
# DatasetMeta Tests
# =============================================================================


def test_dataset_meta_construction() -> None:
    """Verify DatasetMeta can be constructed with required fields."""
    meta = DatasetMeta(
        id="test.dataset",
        name="test.dataset",
        table_key="test.dataset",
        description="Test dataset",
        schema_version="1.0.0",
        family="test",
        is_docs_view=False,
        is_read_only=True,
        default_limit=DEFAULT_LIMIT,
        max_limit=MAX_LIMIT,
    )

    assert meta.id == "test.dataset"
    assert meta.name == "test.dataset"
    assert meta.table_key == "test.dataset"
    assert meta.description == "Test dataset"
    assert meta.schema_version == "1.0.0"
    assert meta.family == "test"
    assert meta.is_docs_view is False
    assert meta.is_read_only is True
    assert meta.default_limit == DEFAULT_LIMIT
    assert meta.max_limit == MAX_LIMIT


def test_dataset_meta_optional_fields() -> None:
    """Verify DatasetMeta optional fields default to None."""
    meta = DatasetMeta(
        id="minimal",
        name="minimal",
        table_key="minimal",
        description="Minimal",
        schema_version=None,
        family=None,
        is_docs_view=False,
        is_read_only=False,
        default_limit=SMALL_DEFAULT_LIMIT,
        max_limit=SMALL_MAX_LIMIT,
    )

    assert meta.owner is None
    assert meta.freshness_sla is None
    assert meta.retention_policy is None
    assert meta.validation_profile is None


def test_dataset_meta_with_all_optional_fields() -> None:
    """Verify DatasetMeta with all optional fields set."""
    meta = DatasetMeta(
        id="full",
        name="full",
        table_key="full",
        description="Full metadata",
        schema_version="2.0",
        family="analytics",
        is_docs_view=True,
        is_read_only=True,
        default_limit=DEFAULT_LIMIT,
        max_limit=MAX_LIMIT,
        owner="analytics-team",
        freshness_sla="1h",
        retention_policy="30d",
        validation_profile="strict",
    )

    assert meta.owner == "analytics-team"
    assert meta.freshness_sla == "1h"
    assert meta.retention_policy == "30d"
    assert meta.validation_profile == "strict"


def test_dataset_meta_is_frozen() -> None:
    """Verify DatasetMeta is a frozen dataclass."""
    meta = DatasetMeta(
        id="frozen",
        name="frozen",
        table_key="frozen",
        description="Frozen",
        schema_version=None,
        family=None,
        is_docs_view=False,
        is_read_only=False,
        default_limit=SMALL_DEFAULT_LIMIT,
        max_limit=SMALL_MAX_LIMIT,
    )

    # Verify the dataclass is frozen by checking the __dataclass_fields__
    assert dataclasses.is_dataclass(meta)
    # A frozen dataclass should have frozen=True in its __dataclass_params__
    params = getattr(type(meta), "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is True
