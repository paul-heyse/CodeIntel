"""Tests for codeintel.config.datasets.dataflow module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.datasets.dataflow import (
    DataflowEdge,
    DataflowNode,
    build_contract_dataflow_graph,
    iter_composite_edges,
    iter_dataset_nodes,
    iter_dependency_edges,
)

if TYPE_CHECKING:
    from codeintel.config.datasets.dataflow import (
        EdgeType,
        NodeKind,
    )


def require(condition: object, message: str) -> None:
    """Fail the current test with a descriptive message."""
    if not condition:
        pytest.fail(message)


def test_dataflow_node_creation() -> None:
    """Verify DataflowNode dataclass behaves correctly."""
    node = DataflowNode(
        id="test.table",
        kind="table",
        family="test",
        owner_package="core",
        description="Test table",
    )
    require(node.id == "test.table", "id should store provided value")
    require(node.kind == "table", "kind should store provided value")
    require(node.family == "test", "family should store provided value")
    require(node.owner_package == "core", "owner_package should store provided value")
    require(node.description == "Test table", "description should store provided value")


def test_dataflow_node_defaults() -> None:
    """Verify DataflowNode has correct default values."""
    node = DataflowNode(id="test.node", kind="table")
    require(node.family is None, "family should default to None")
    require(node.owner_package is None, "owner_package should default to None")
    require(node.description is None, "description should default to None")


def test_dataflow_edge_creation() -> None:
    """Verify DataflowEdge dataclass behaves correctly."""
    edge = DataflowEdge(src="source.table", dst="target.table", edge_type="builds")
    require(edge.src == "source.table", "src should store provided value")
    require(edge.dst == "target.table", "dst should store provided value")
    require(edge.edge_type == "builds", "edge_type should store provided value")


def test_node_kind_literal() -> None:
    """Verify NodeKind literal values are valid."""
    valid_kinds: list[NodeKind] = ["table", "view", "operation", "graph"]
    for kind in valid_kinds:
        node = DataflowNode(id="test", kind=kind)
        require(node.kind == kind, "kind should store provided literal")


def test_edge_type_literal() -> None:
    """Verify EdgeType literal values are valid."""
    valid_types: list[EdgeType] = ["builds", "reads", "exposes", "depends_on"]
    for edge_type in valid_types:
        edge = DataflowEdge(src="a", dst="b", edge_type=edge_type)
        require(edge.edge_type == edge_type, "edge_type should store provided literal")


def test_iter_dataset_nodes_returns_iterator() -> None:
    """Verify iter_dataset_nodes returns an iterator of DataflowNodes."""
    nodes = list(iter_dataset_nodes())
    require(len(nodes) > 0, "dataset nodes iterator should yield values")
    for node in nodes:
        require(isinstance(node, DataflowNode), "iterator should yield DataflowNode")
        require(node.kind in {"table", "view"}, "node.kind should be table or view")


def test_iter_composite_edges_returns_iterator() -> None:
    """Verify iter_composite_edges returns an iterator of DataflowEdges."""
    edges = list(iter_composite_edges())
    # May be empty if no composite schemas, but should be iterable
    for edge in edges:
        require(isinstance(edge, DataflowEdge), "iterator should yield DataflowEdge")
        require(edge.edge_type == "builds", "composite edges should be type builds")


def test_iter_dependency_edges_returns_iterator() -> None:
    """Verify iter_dependency_edges returns an iterator of DataflowEdges."""
    edges = list(iter_dependency_edges())
    # May be empty if no dependencies, but should be iterable
    for edge in edges:
        require(isinstance(edge, DataflowEdge), "iterator should yield DataflowEdge")
        require(edge.edge_type == "builds", "dependency edges should be type builds")


def test_build_contract_dataflow_graph_returns_tuple() -> None:
    """Verify build_contract_dataflow_graph returns nodes and edges."""
    nodes, edges = build_contract_dataflow_graph()
    require(isinstance(nodes, list), "nodes should be a list")
    require(isinstance(edges, list), "edges should be a list")
    require(len(nodes) > 0, "dataflow graph should include dataset nodes")


def test_build_contract_dataflow_graph_nodes_are_unique() -> None:
    """Verify all nodes in the dataflow graph have unique IDs."""
    nodes, _ = build_contract_dataflow_graph()
    node_ids = [node.id for node in nodes]
    require(
        len(node_ids) == len(set(node_ids)),
        "duplicate node IDs found in dataflow graph",
    )


def test_build_contract_dataflow_graph_edges_are_unique() -> None:
    """Verify all edges in the dataflow graph are unique."""
    _, edges = build_contract_dataflow_graph()
    edge_keys = [(e.src, e.dst, e.edge_type) for e in edges]
    require(
        len(edge_keys) == len(set(edge_keys)),
        "duplicate edges found in dataflow graph",
    )


def test_dataflow_graph_contains_expected_tables() -> None:
    """Verify the dataflow graph contains known dataset tables."""
    nodes, _ = build_contract_dataflow_graph()
    node_ids = {node.id for node in nodes}

    # Check for some known tables
    require("core.goids" in node_ids, "core.goids should appear in dataflow graph")
    require(
        "analytics.function_metrics" in node_ids,
        "analytics.function_metrics should appear in dataflow graph",
    )
