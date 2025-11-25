"""Unit tests for graph_service helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import networkx as nx
import pytest

from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    centrality_undirected,
    component_metadata,
    neighbor_stats,
    projection_metrics,
    structural_metrics,
)


def test_centrality_directed_weighted_and_seeded() -> None:
    """Directed centrality respects weights and seeded randomness."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(2, 3, weight=1.0)
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC), seed=42)
    metrics = centrality_directed(graph, ctx)
    if metrics.betweenness[2] <= metrics.betweenness.get(1, 0.0):
        pytest.fail("Expected node 2 betweenness to exceed node 1")
    if metrics.pagerank[1] == metrics.pagerank[2]:
        pytest.fail("Expected pagerank variation across nodes")


def test_centrality_undirected_empty_returns_zeroes() -> None:
    """Undirected centrality returns empty metrics for empty graph."""
    graph = nx.Graph()
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC))
    metrics = centrality_undirected(graph, ctx)
    if metrics.betweenness or metrics.pagerank:
        pytest.fail("Expected empty centrality metrics for empty graph")


def test_neighbor_stats_with_weights() -> None:
    """Neighbor stats aggregate weighted degrees."""
    graph = nx.DiGraph()
    weighted_edge_value = 3
    graph.add_edge("a", "b", weight=weighted_edge_value)
    stats = neighbor_stats(graph, weight="weight")
    if stats.out_counts.get("a") != weighted_edge_value:
        pytest.fail(
            f"Expected weighted out count of {weighted_edge_value}, got {stats.out_counts.get('a')}"
        )
    if "b" not in stats.in_neighbors:
        pytest.fail("Expected inbound neighbor entry for 'b'")


def test_component_metadata_cycles_and_layers() -> None:
    """Component metadata marks cycles and assigns layers."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 1), (2, 3)])
    meta = component_metadata(graph)
    if not meta.in_cycle.get(1):
        pytest.fail("Node 1 should be marked as part of a cycle")
    if meta.layer.get(3, 0) < meta.layer.get(1, 0):
        pytest.fail("Downstream node should not have lower layer than cycle node")


def test_projection_metrics_basic() -> None:
    """Projection metrics compute weighted degree for projected nodes."""
    graph = nx.Graph()
    graph.add_edge(("x", 1), ("y", 1), weight=1.0)
    graph.add_edge(("x", 2), ("y", 1), weight=2.0)
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC), seed=0)
    metrics = projection_metrics(graph, {("x", 1), ("x", 2)}, ctx)
    weighted_degree = metrics.weighted_degree.get(("x", 1), 0.0)
    if weighted_degree <= 0.0:
        pytest.fail(f"Expected projected weighted degree to be positive, got {weighted_degree}")


def test_structural_metrics_triangle_and_core() -> None:
    """Structural metrics include triangle counts and k-core values."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    metrics = structural_metrics(graph)
    expected_triangles = 1
    if metrics.triangles.get(1) != expected_triangles:
        pytest.fail(f"Unexpected triangle count: {metrics.triangles.get(1)}")
    if metrics.core_number.get(1, 0) < 1:
        pytest.fail("Expected node 1 to be part of core >= 1")
