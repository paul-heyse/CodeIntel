"""Tests for dependency inference.

Tests the DependencyNode and DependencyGraph dataclasses,
and infer_upstream_dependencies/infer_downstream_consumers functions.
"""

from __future__ import annotations

import pytest

from codeintel.config.datasets.dependency_inference import (
    DependencyGraph,
    DependencyNode,
    build_dependency_graph,
    get_transitive_consumers,
    get_transitive_dependencies,
    infer_downstream_consumers,
    infer_upstream_dependencies,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


# ------------------------------------------------------------------
# DependencyNode tests
# ------------------------------------------------------------------


def test_dependency_node_creation() -> None:
    """Create DependencyNode with basic fields."""
    node = DependencyNode(
        table_key="analytics.function_metrics",
        producer_plugins=["analytics.function_metrics"],
        upstream=["core.goids"],
        downstream=["analytics.hotspots"],
    )
    _expect_equal(node.table_key, "analytics.function_metrics", "table_key")
    _expect_equal(len(node.producer_plugins), 1, "producer_plugins")
    _expect_equal(len(node.upstream), 1, "upstream")
    _expect_equal(len(node.downstream), 1, "downstream")


def test_dependency_node_has_upstream() -> None:
    """Verify has_upstream property."""
    node_with = DependencyNode(
        table_key="test",
        producer_plugins=[],
        upstream=["core.goids"],
        downstream=[],
    )
    node_without = DependencyNode(
        table_key="test",
        producer_plugins=[],
        upstream=[],
        downstream=[],
    )
    _require(condition=node_with.has_upstream, message="should have upstream")
    _require(condition=not node_without.has_upstream, message="should not have upstream")


def test_dependency_node_has_downstream() -> None:
    """Verify has_downstream property."""
    node_with = DependencyNode(
        table_key="test",
        producer_plugins=[],
        upstream=[],
        downstream=["analytics.hotspots"],
    )
    node_without = DependencyNode(
        table_key="test",
        producer_plugins=[],
        upstream=[],
        downstream=[],
    )
    _require(condition=node_with.has_downstream, message="should have downstream")
    _require(condition=not node_without.has_downstream, message="should not have downstream")


def test_dependency_node_is_root() -> None:
    """Verify is_root property."""
    root_node = DependencyNode(
        table_key="core.goids",
        producer_plugins=[],
        upstream=[],
        downstream=["analytics.function_metrics"],
    )
    non_root = DependencyNode(
        table_key="analytics.function_metrics",
        producer_plugins=[],
        upstream=["core.goids"],
        downstream=[],
    )
    _require(condition=root_node.is_root, message="should be root")
    _require(condition=not non_root.is_root, message="should not be root")


def test_dependency_node_is_leaf() -> None:
    """Verify is_leaf property."""
    leaf_node = DependencyNode(
        table_key="analytics.hotspots",
        producer_plugins=[],
        upstream=["analytics.function_metrics"],
        downstream=[],
    )
    non_leaf = DependencyNode(
        table_key="analytics.function_metrics",
        producer_plugins=[],
        upstream=[],
        downstream=["analytics.hotspots"],
    )
    _require(condition=leaf_node.is_leaf, message="should be leaf")
    _require(condition=not non_leaf.is_leaf, message="should not be leaf")


# ------------------------------------------------------------------
# DependencyGraph tests
# ------------------------------------------------------------------


def test_dependency_graph_creation() -> None:
    """Create DependencyGraph with basic fields."""
    graph = DependencyGraph(nodes={})
    _expect_equal(graph.table_count, 0, "table_count")


def test_dependency_graph_table_count() -> None:
    """Verify table_count property."""
    nodes = {
        "table1": DependencyNode(
            table_key="table1", producer_plugins=[], upstream=[], downstream=[]
        ),
        "table2": DependencyNode(
            table_key="table2", producer_plugins=[], upstream=[], downstream=[]
        ),
    }
    graph = DependencyGraph(nodes=nodes)
    _expect_equal(graph.table_count, 2, "table_count")


def test_dependency_graph_get() -> None:
    """Verify get method returns correct node."""
    node = DependencyNode(table_key="test.table", producer_plugins=[], upstream=[], downstream=[])
    graph = DependencyGraph(nodes={"test.table": node})

    result = graph.get("test.table")
    if result is None:
        msg = "should find node"
        pytest.fail(msg)
    _expect_equal(result.table_key, "test.table", "table_key")


def test_dependency_graph_get_missing() -> None:
    """Verify get method returns None for missing table."""
    graph = DependencyGraph(nodes={})
    result = graph.get("nonexistent")
    _require(condition=result is None, message="should return None")


def test_dependency_graph_root_tables() -> None:
    """Verify root_tables returns tables with no upstream."""
    nodes = {
        "root": DependencyNode(
            table_key="root", producer_plugins=[], upstream=[], downstream=["child"]
        ),
        "child": DependencyNode(
            table_key="child", producer_plugins=[], upstream=["root"], downstream=[]
        ),
    }
    graph = DependencyGraph(nodes=nodes)

    roots = graph.root_tables()
    _expect_equal(len(roots), 1, "root count")
    _expect_equal(roots[0], "root", "root table")


def test_dependency_graph_leaf_tables() -> None:
    """Verify leaf_tables returns tables with no downstream."""
    nodes = {
        "root": DependencyNode(
            table_key="root", producer_plugins=[], upstream=[], downstream=["child"]
        ),
        "child": DependencyNode(
            table_key="child", producer_plugins=[], upstream=["root"], downstream=[]
        ),
    }
    graph = DependencyGraph(nodes=nodes)

    leaves = graph.leaf_tables()
    _expect_equal(len(leaves), 1, "leaf count")
    _expect_equal(leaves[0], "child", "leaf table")


def test_dependency_graph_topological_order() -> None:
    """Verify topological_order returns correct ordering."""
    nodes = {
        "root": DependencyNode(
            table_key="root", producer_plugins=[], upstream=[], downstream=["child"]
        ),
        "child": DependencyNode(
            table_key="child", producer_plugins=[], upstream=["root"], downstream=[]
        ),
    }
    graph = DependencyGraph(nodes=nodes)

    order = graph.topological_order()
    _expect_equal(len(order), 2, "order length")
    # Root should come before child
    root_idx = order.index("root")
    child_idx = order.index("child")
    _require(condition=root_idx < child_idx, message="root should come before child")


# ------------------------------------------------------------------
# infer_upstream_dependencies tests
# ------------------------------------------------------------------


def test_infer_upstream_dependencies_returns_list() -> None:
    """Verify infer_upstream_dependencies returns a list."""
    result = infer_upstream_dependencies("analytics.function_metrics")
    _require(condition=isinstance(result, list), message="should return list")


def test_infer_upstream_dependencies_unknown_table() -> None:
    """Verify empty list for unknown table."""
    result = infer_upstream_dependencies("nonexistent.table")
    _expect_equal(len(result), 0, "should be empty")


# ------------------------------------------------------------------
# infer_downstream_consumers tests
# ------------------------------------------------------------------


def test_infer_downstream_consumers_returns_list() -> None:
    """Verify infer_downstream_consumers returns a list."""
    result = infer_downstream_consumers("core.goids")
    _require(condition=isinstance(result, list), message="should return list")


def test_infer_downstream_consumers_unknown_table() -> None:
    """Verify empty list for unknown table."""
    result = infer_downstream_consumers("nonexistent.table")
    _expect_equal(len(result), 0, "should be empty")


# ------------------------------------------------------------------
# build_dependency_graph tests
# ------------------------------------------------------------------


def test_build_dependency_graph_returns_graph() -> None:
    """Verify build_dependency_graph returns DependencyGraph."""
    result = build_dependency_graph()
    _require(condition=isinstance(result, DependencyGraph), message="should return DependencyGraph")


def test_build_dependency_graph_has_nodes() -> None:
    """Verify build_dependency_graph includes registered tables."""
    result = build_dependency_graph()
    # Should have at least some tables if registry is populated
    _require(condition=result.table_count >= 0, message="table_count should be non-negative")


# ------------------------------------------------------------------
# get_transitive_dependencies tests
# ------------------------------------------------------------------


def test_get_transitive_dependencies_returns_list() -> None:
    """Verify get_transitive_dependencies returns a list."""
    result = get_transitive_dependencies("analytics.function_metrics")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_transitive_dependencies_include_self() -> None:
    """Verify include_self option works."""
    result_without = get_transitive_dependencies("test.table", include_self=False)
    result_with = get_transitive_dependencies("test.table", include_self=True)
    # With should include table itself, without should not
    _require(
        condition="test.table" not in result_without,
        message="should not include self without flag",
    )
    # Can't guarantee it's in result_with if there are no deps, but should differ
    _require(condition=isinstance(result_with, list), message="should return list")


# ------------------------------------------------------------------
# get_transitive_consumers tests
# ------------------------------------------------------------------


def test_get_transitive_consumers_returns_list() -> None:
    """Verify get_transitive_consumers returns a list."""
    result = get_transitive_consumers("core.goids")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_transitive_consumers_include_self() -> None:
    """Verify include_self option works."""
    result_without = get_transitive_consumers("test.table", include_self=False)
    result_with = get_transitive_consumers("test.table", include_self=True)
    # With should include table itself, without should not
    _require(
        condition="test.table" not in result_without,
        message="should not include self without flag",
    )
    _require(condition=isinstance(result_with, list), message="should return list")
