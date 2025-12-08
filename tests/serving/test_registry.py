"""Tests for serving layer registry module.

This module tests the dataset registry and dataflow graph building functionality
using real DATASET_CONTRACTS and operations from the canonical catalogs.
"""

from __future__ import annotations

import dataclasses

import pytest

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.operations.catalog import (
    DatasetMeta,
    build_serving_dataflow_graph,
    get_registry_operation,
    iter_graph_nodes,
    iter_operation_dataset_edges,
    iter_operation_graph_edges,
    iter_operation_nodes,
    iter_registry_operations,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
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

    expect_is_instance(operations, list)
    expect_true(len(operations) > 0)


def test_iter_registry_operations_all_have_ids() -> None:
    """Verify all operations have non-empty IDs."""
    operations = iter_registry_operations()

    for op in operations:
        expect_true(hasattr(op, "id"))
        expect_true(bool(op.id))
        expect_is_instance(op.id, str)


def test_iter_registry_operations_includes_datasets_rows() -> None:
    """Verify datasets.rows operation exists and has exposed_datasets patched."""
    operations = iter_registry_operations()
    datasets_rows = next((op for op in operations if op.id == "datasets.rows"), None)

    if datasets_rows is None:
        pytest.fail("datasets.rows operation missing")
    expect_is_not_none(datasets_rows)
    expect_true(hasattr(datasets_rows, "exposed_datasets"))
    # Should be patched with DATASET_CONTRACTS_BY_TABLE_KEY keys
    expect_true(len(datasets_rows.exposed_datasets) > 0)


def test_iter_registry_operations_patched_datasets_match_contracts() -> None:
    """Verify datasets.rows exposed_datasets match actual contract keys."""
    operations = iter_registry_operations()
    datasets_rows = next((op for op in operations if op.id == "datasets.rows"), None)

    if datasets_rows is None:
        pytest.fail("datasets.rows operation missing")
    expect_is_not_none(datasets_rows)
    exposed_set = set(datasets_rows.exposed_datasets)
    contract_keys = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    # Exposed datasets should be exactly the contract keys
    expect_equal(exposed_set, contract_keys)


# =============================================================================
# get_registry_operation Tests
# =============================================================================


def test_get_registry_operation_found() -> None:
    """Verify get_registry_operation returns operation by ID."""
    operations = iter_registry_operations()
    # Pick first operation ID
    first_op = operations[0]

    result = get_registry_operation(first_op.id)

    if result is None:
        pytest.fail("get_registry_operation returned None")
    expect_is_not_none(result)
    expect_equal(result.id, first_op.id)


def test_get_registry_operation_not_found() -> None:
    """Verify get_registry_operation returns None for unknown ID."""
    result = get_registry_operation("non_existent_operation_xyz")

    expect_true(result is None)


def test_get_registry_operation_datasets_rows() -> None:
    """Verify datasets.rows can be retrieved."""
    result = get_registry_operation("datasets.rows")

    if result is None:
        pytest.fail("datasets.rows operation missing")
    expect_is_not_none(result)
    expect_equal(result.id, "datasets.rows")
    expect_true(hasattr(result, "exposed_datasets"))


# =============================================================================
# iter_operation_nodes Tests
# =============================================================================


def test_iter_operation_nodes_returns_list() -> None:
    """Verify iter_operation_nodes returns DataflowNode list."""
    nodes = iter_operation_nodes()

    expect_is_instance(nodes, list)
    expect_true(len(nodes) > 0)


def test_iter_operation_nodes_all_dataflow_nodes() -> None:
    """Verify all returned items are DataflowNode instances."""
    nodes = iter_operation_nodes()

    for node in nodes:
        expect_is_instance(node, DataflowNode)


def test_iter_operation_nodes_have_operation_kind() -> None:
    """Verify all operation nodes have kind='operation'."""
    nodes = iter_operation_nodes()

    for node in nodes:
        expect_equal(node.kind, "operation")


def test_iter_operation_nodes_have_serving_family() -> None:
    """Verify all operation nodes have family='serving'."""
    nodes = iter_operation_nodes()

    for node in nodes:
        expect_equal(node.family, "serving")


def test_iter_operation_nodes_match_operations() -> None:
    """Verify operation nodes match registered operations."""
    operations = iter_registry_operations()
    nodes = iter_operation_nodes()

    op_ids = {op.id for op in operations}
    node_ids = {node.id for node in nodes}

    expect_equal(op_ids, node_ids)


# =============================================================================
# iter_graph_nodes Tests
# =============================================================================


def test_iter_graph_nodes_returns_list() -> None:
    """Verify iter_graph_nodes returns DataflowNode list."""
    nodes = iter_graph_nodes()

    expect_is_instance(nodes, list)
    # May be empty if no operations require graphs


def test_iter_graph_nodes_all_dataflow_nodes() -> None:
    """Verify all returned items are DataflowNode instances."""
    nodes = iter_graph_nodes()

    for node in nodes:
        expect_is_instance(node, DataflowNode)


def test_iter_graph_nodes_have_graph_kind() -> None:
    """Verify all graph nodes have kind='graph'."""
    nodes = iter_graph_nodes()

    for node in nodes:
        expect_equal(node.kind, "graph")


def test_iter_graph_nodes_have_graph_prefix_ids() -> None:
    """Verify all graph node IDs start with 'graph.'."""
    nodes = iter_graph_nodes()

    for node in nodes:
        expect_true(node.id.startswith("graph."))


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

    expect_equal(node_ids, required_graphs)


# =============================================================================
# iter_operation_dataset_edges Tests
# =============================================================================


def test_iter_operation_dataset_edges_returns_list() -> None:
    """Verify iter_operation_dataset_edges returns DataflowEdge list."""
    edges = iter_operation_dataset_edges()

    expect_is_instance(edges, list)


def test_iter_operation_dataset_edges_all_dataflow_edges() -> None:
    """Verify all returned items are DataflowEdge instances."""
    edges = iter_operation_dataset_edges()

    for edge in edges:
        expect_is_instance(edge, DataflowEdge)


def test_iter_operation_dataset_edges_have_valid_types() -> None:
    """Verify all edges have 'reads' or 'exposes' edge_type."""
    edges = iter_operation_dataset_edges()

    valid_types = {"reads", "exposes"}
    for edge in edges:
        expect_in(edge.edge_type, valid_types)


def test_iter_operation_dataset_edges_dst_are_operations() -> None:
    """Verify all edge destinations are valid operation IDs."""
    operations = iter_registry_operations()
    edges = iter_operation_dataset_edges()

    op_ids = {op.id for op in operations}
    for edge in edges:
        expect_in(edge.dst, op_ids)


# =============================================================================
# iter_operation_graph_edges Tests
# =============================================================================


def test_iter_operation_graph_edges_returns_list() -> None:
    """Verify iter_operation_graph_edges returns DataflowEdge list."""
    edges = iter_operation_graph_edges()

    expect_is_instance(edges, list)


def test_iter_operation_graph_edges_all_dataflow_edges() -> None:
    """Verify all returned items are DataflowEdge instances."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        expect_is_instance(edge, DataflowEdge)


def test_iter_operation_graph_edges_have_depends_on_type() -> None:
    """Verify all graph edges have 'depends_on' edge_type."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        expect_equal(edge.edge_type, "depends_on")


def test_iter_operation_graph_edges_src_are_graph_nodes() -> None:
    """Verify all edge sources are graph node IDs."""
    edges = iter_operation_graph_edges()

    for edge in edges:
        expect_true(edge.src.startswith("graph."))


def test_iter_operation_graph_edges_dst_are_operations() -> None:
    """Verify all edge destinations are valid operation IDs."""
    operations = iter_registry_operations()
    edges = iter_operation_graph_edges()

    op_ids = {op.id for op in operations}
    for edge in edges:
        expect_in(edge.dst, op_ids)


# =============================================================================
# build_serving_dataflow_graph Tests
# =============================================================================


def test_build_serving_dataflow_graph_returns_tuple() -> None:
    """Verify build_serving_dataflow_graph returns nodes and edges tuple."""
    result = build_serving_dataflow_graph()

    expect_is_instance(result, tuple)
    expect_equal(len(result), TUPLE_LENGTH)


def test_build_serving_dataflow_graph_nodes_list() -> None:
    """Verify first element is list of DataflowNode."""
    nodes, _ = build_serving_dataflow_graph()

    expect_is_instance(nodes, list)
    for node in nodes:
        expect_is_instance(node, DataflowNode)


def test_build_serving_dataflow_graph_edges_list() -> None:
    """Verify second element is list of DataflowEdge."""
    _, edges = build_serving_dataflow_graph()

    expect_is_instance(edges, list)
    for edge in edges:
        expect_is_instance(edge, DataflowEdge)


def test_build_serving_dataflow_graph_includes_operations() -> None:
    """Verify graph includes operation nodes."""
    nodes, _ = build_serving_dataflow_graph()

    operation_nodes = [n for n in nodes if n.kind == "operation"]
    expect_true(len(operation_nodes) > 0)


def test_build_serving_dataflow_graph_includes_datasets() -> None:
    """Verify graph includes dataset nodes."""
    nodes, _ = build_serving_dataflow_graph()

    # Dataset nodes have kind 'dataset' or 'view'
    dataset_view_kinds = {"dataset", "view"}
    dataset_nodes = [n for n in nodes if n.kind in dataset_view_kinds]
    expect_true(len(dataset_nodes) > 0)


def test_build_serving_dataflow_graph_edges_deduplicated() -> None:
    """Verify edges are deduplicated."""
    _, edges = build_serving_dataflow_graph()

    edge_keys = [(e.src, e.dst, e.edge_type) for e in edges]
    unique_keys = set(edge_keys)

    expect_equal(len(edge_keys), len(unique_keys))


def test_build_serving_dataflow_graph_nodes_deduplicated() -> None:
    """Verify nodes are deduplicated by (id, kind)."""
    nodes, _ = build_serving_dataflow_graph()

    node_keys = [(n.id, n.kind) for n in nodes]
    unique_keys = set(node_keys)

    expect_equal(len(node_keys), len(unique_keys))


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

    expect_equal(meta.id, "test.dataset")
    expect_equal(meta.name, "test.dataset")
    expect_equal(meta.table_key, "test.dataset")
    expect_equal(meta.description, "Test dataset")
    expect_equal(meta.schema_version, "1.0.0")
    expect_equal(meta.family, "test")
    expect_false(meta.is_docs_view)
    expect_true(meta.is_read_only)
    expect_equal(meta.default_limit, DEFAULT_LIMIT)
    expect_equal(meta.max_limit, MAX_LIMIT)


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

    expect_true(meta.owner is None)
    expect_true(meta.freshness_sla is None)
    expect_true(meta.retention_policy is None)
    expect_true(meta.validation_profile is None)


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

    expect_equal(meta.owner, "analytics-team")
    expect_equal(meta.freshness_sla, "1h")
    expect_equal(meta.retention_policy, "30d")
    expect_equal(meta.validation_profile, "strict")


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
    expect_true(dataclasses.is_dataclass(meta))
    # A frozen dataclass should have frozen=True in its __dataclass_params__
    params = getattr(type(meta), "__dataclass_params__", None)
    if params is None:
        pytest.fail("DatasetMeta dataclass params missing")
    expect_is_not_none(params)
    expect_true(params.frozen is True)
