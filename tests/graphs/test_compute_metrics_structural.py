"""Tests for structural metric computation functions.

This module tests the stateless structural computation functions including
clustering coefficient, triangles, k-core, constraint, and effective size.
"""

from __future__ import annotations

import math
from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.structural import (
    StructuralMetrics,
    compute_all_structural,
    compute_clustering_coefficient,
    compute_constraint,
    compute_core_number,
    compute_effective_size,
    compute_triangles,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    diamond_graph,
    star_graph,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
TOLERANCE: Final[float] = 0.01
CLUSTERING_ZERO: Final[float] = 0.0
CLUSTERING_ONE: Final[float] = 1.0
TRIANGLES_ZERO: Final[int] = 0
TRIANGLES_ONE: Final[int] = 1
TRIANGLES_K4: Final[int] = 3  # C(3,2) triangles per node in K4
CORE_NUMBER_ZERO: Final[int] = 0
CORE_NUMBER_ONE: Final[int] = 1
CORE_NUMBER_TWO: Final[int] = 2
CORE_NUMBER_K4: Final[int] = 3  # K4 is a 3-core
CONSTRAINT_ZERO: Final[float] = 0.0


# ===========================================================================
# compute_clustering_coefficient Tests
# ===========================================================================


def test_clustering_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_clustering_coefficient(graph)
    expect_equal(result, {})


def test_clustering_single_node() -> None:
    """Single node has clustering 0."""
    graph = nx.Graph()
    graph.add_node("A")
    result = compute_clustering_coefficient(graph)

    expect_length(result, 1)
    expect_equal(result["A"], CLUSTERING_ZERO)


def test_clustering_chain_graph() -> None:
    """Chain graph has zero clustering (no triangles)."""
    graph = chain_graph(5).to_undirected()
    result = compute_clustering_coefficient(graph)

    for clustering in result.values():
        expect_equal(clustering, CLUSTERING_ZERO)


def test_clustering_complete_graph() -> None:
    """Complete graph has clustering coefficient 1.0."""
    graph = nx.complete_graph(5)
    result = compute_clustering_coefficient(graph)

    for clustering in result.values():
        expect_true(abs(clustering - CLUSTERING_ONE) < TOLERANCE)


def test_clustering_triangle() -> None:
    """Triangle graph has clustering 1.0 for all nodes."""
    graph = nx.cycle_graph(3)  # Triangle
    result = compute_clustering_coefficient(graph)

    for clustering in result.values():
        expect_true(abs(clustering - CLUSTERING_ONE) < TOLERANCE)


def test_clustering_star_graph() -> None:
    """Star graph center has clustering 0."""
    graph = star_graph(4).to_undirected()
    result = compute_clustering_coefficient(graph)

    # Hub has no triangles (spokes not connected to each other)
    expect_equal(result["hub"], CLUSTERING_ZERO)
    # Spokes also have clustering 0 (only connected to hub)
    for i in range(1, 5):
        expect_equal(result[f"spoke{i}"], CLUSTERING_ZERO)


def test_clustering_directed_graph_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = nx.complete_graph(4, create_using=nx.DiGraph())
    result = compute_clustering_coefficient(graph)

    # Should return results (converted to undirected)
    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_clustering_diamond_graph() -> None:
    """Diamond graph has mixed clustering."""
    graph = diamond_graph().to_undirected()
    result = compute_clustering_coefficient(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


# ===========================================================================
# compute_triangles Tests
# ===========================================================================


def test_triangles_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_triangles(graph)
    expect_equal(result, {})


def test_triangles_single_node() -> None:
    """Single node has 0 triangles."""
    graph = nx.Graph()
    graph.add_node("A")
    result = compute_triangles(graph)

    expect_equal(result["A"], TRIANGLES_ZERO)


def test_triangles_chain_graph() -> None:
    """Chain graph has no triangles."""
    graph = chain_graph(5).to_undirected()
    result = compute_triangles(graph)

    for count in result.values():
        expect_equal(count, TRIANGLES_ZERO)


def test_triangles_single_triangle() -> None:
    """Triangle graph: each node participates in 1 triangle."""
    graph = nx.cycle_graph(3)
    result = compute_triangles(graph)

    for count in result.values():
        expect_equal(count, TRIANGLES_ONE)


def test_triangles_complete_graph() -> None:
    """Complete graph K4 has multiple triangles per node."""
    graph = nx.complete_graph(4)
    result = compute_triangles(graph)

    # In K4, each node participates in C(3,2) = 3 triangles
    for count in result.values():
        expect_equal(count, TRIANGLES_K4)


def test_triangles_star_graph() -> None:
    """Star graph has no triangles."""
    graph = star_graph(5).to_undirected()
    result = compute_triangles(graph)

    for count in result.values():
        expect_equal(count, TRIANGLES_ZERO)


def test_triangles_directed_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    result = compute_triangles(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)


# ===========================================================================
# compute_core_number Tests
# ===========================================================================


def test_core_number_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_core_number(graph)
    expect_equal(result, {})


def test_core_number_single_node() -> None:
    """Single node has core number 0."""
    graph = nx.Graph()
    graph.add_node("A")
    result = compute_core_number(graph)

    expect_equal(result["A"], CORE_NUMBER_ZERO)


def test_core_number_chain_graph() -> None:
    """Chain graph: ends have core 1, middles have core 1."""
    graph = chain_graph(5).to_undirected()
    result = compute_core_number(graph)

    # All nodes in chain have core number 1 (each has at least 1 neighbor)
    for core in result.values():
        expect_equal(core, CORE_NUMBER_ONE)


def test_core_number_complete_graph() -> None:
    """Complete graph K4: all nodes have core number n-1."""
    graph = nx.complete_graph(4)
    result = compute_core_number(graph)

    # K4 is a 3-core (each node has 3 neighbors)
    for core in result.values():
        expect_equal(core, CORE_NUMBER_K4)


def test_core_number_star_graph() -> None:
    """Star graph: all nodes in 1-core."""
    graph = star_graph(5).to_undirected()
    result = compute_core_number(graph)

    # Star graph is 1-core
    for core in result.values():
        expect_equal(core, CORE_NUMBER_ONE)


def test_core_number_triangle() -> None:
    """Triangle: all nodes in 2-core."""
    graph = nx.cycle_graph(3)
    result = compute_core_number(graph)

    for core in result.values():
        expect_equal(core, CORE_NUMBER_TWO)


def test_core_number_directed_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = nx.complete_graph(4, create_using=nx.DiGraph())
    result = compute_core_number(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


# ===========================================================================
# compute_constraint Tests
# ===========================================================================


def test_constraint_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_constraint(graph)
    expect_equal(result, {})


def test_constraint_single_node() -> None:
    """Single node has constraint 0 (no neighbors)."""
    graph = nx.Graph()
    graph.add_node("A")
    result = compute_constraint(graph)

    expect_equal(result["A"], CONSTRAINT_ZERO)


def test_constraint_chain_graph() -> None:
    """Chain graph: ends have high constraint."""
    graph = chain_graph(4).to_undirected()
    result = compute_constraint(graph)

    # All nodes should have some constraint value
    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_constraint_star_graph() -> None:
    """Star graph: hub has low constraint (structural hole)."""
    graph = star_graph(4).to_undirected()
    result = compute_constraint(graph)

    # Hub connects to disconnected spokes - spans structural holes
    # Spokes have high constraint (only connected to hub)
    expect_length(result, EXPECTED_NODE_COUNT_FIVE)


def test_constraint_complete_graph() -> None:
    """Complete graph: all have same constraint (no structural holes)."""
    graph = nx.complete_graph(4)
    result = compute_constraint(graph)

    # All nodes have same constraint in complete graph
    constraints = list(result.values())
    for c in constraints:
        expect_true(abs(c - constraints[0]) < TOLERANCE)


def test_constraint_directed_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = chain_graph(3)
    result = compute_constraint(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)


# ===========================================================================
# compute_effective_size Tests
# ===========================================================================


def test_effective_size_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_effective_size(graph)
    expect_equal(result, {})


def test_effective_size_single_node() -> None:
    """Single node has effective size nan (no neighbors).

    NetworkX returns nan for nodes with no neighbors because effective
    size is undefined without neighbors to be redundant with.
    """
    graph = nx.Graph()
    graph.add_node("A")
    result = compute_effective_size(graph)

    expect_true(math.isnan(result["A"]))


def test_effective_size_star_graph() -> None:
    """Star graph: hub has high effective size."""
    graph = star_graph(4).to_undirected()
    result = compute_effective_size(graph)

    # Hub's ego network is non-redundant (spokes not connected)
    # Effective size ≈ degree
    expect_true(result["hub"] > 0)


def test_effective_size_complete_graph() -> None:
    """Complete graph: low effective size (redundant connections)."""
    graph = nx.complete_graph(4)
    result = compute_effective_size(graph)

    # All neighbors are connected, so redundancy is high
    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_effective_size_chain_graph() -> None:
    """Chain graph effective size."""
    graph = chain_graph(4).to_undirected()
    result = compute_effective_size(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_effective_size_directed_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = chain_graph(3)
    result = compute_effective_size(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)


# ===========================================================================
# compute_all_structural Tests
# ===========================================================================


def test_all_structural_empty_graph() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_all_structural(graph)
    expect_equal(result, {})


def test_all_structural_returns_dataclass() -> None:
    """Returns StructuralMetrics dataclass for each node."""
    graph = nx.complete_graph(4)
    result = compute_all_structural(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    for metrics in result.values():
        expect_true(isinstance(metrics, StructuralMetrics))
        expect_true(hasattr(metrics, "clustering"))
        expect_true(hasattr(metrics, "triangles"))
        expect_true(hasattr(metrics, "core_number"))
        expect_true(hasattr(metrics, "constraint"))
        expect_true(hasattr(metrics, "effective_size"))


def test_all_structural_chain_graph() -> None:
    """Chain graph structural metrics."""
    graph = chain_graph(4).to_undirected()
    result = compute_all_structural(graph)

    for metrics in result.values():
        expect_equal(metrics.clustering, CLUSTERING_ZERO)
        expect_equal(metrics.triangles, TRIANGLES_ZERO)
        expect_equal(metrics.core_number, CORE_NUMBER_ONE)


def test_all_structural_complete_graph() -> None:
    """Complete graph structural metrics."""
    graph = nx.complete_graph(4)
    result = compute_all_structural(graph)

    for metrics in result.values():
        expect_true(abs(metrics.clustering - CLUSTERING_ONE) < TOLERANCE)
        expect_equal(metrics.triangles, TRIANGLES_K4)  # C(3,2)
        expect_equal(metrics.core_number, CORE_NUMBER_K4)  # K4 is 3-core


def test_all_structural_star_graph() -> None:
    """Star graph structural metrics."""
    graph = star_graph(4).to_undirected()
    result = compute_all_structural(graph)

    # Hub
    expect_equal(result["hub"].clustering, CLUSTERING_ZERO)
    expect_equal(result["hub"].triangles, TRIANGLES_ZERO)
    expect_equal(result["hub"].core_number, CORE_NUMBER_ONE)

    # Spokes
    for i in range(1, 5):
        expect_equal(result[f"spoke{i}"].clustering, CLUSTERING_ZERO)
        expect_equal(result[f"spoke{i}"].triangles, TRIANGLES_ZERO)


def test_all_structural_triangle() -> None:
    """Triangle structural metrics."""
    graph = nx.cycle_graph(3)
    result = compute_all_structural(graph)

    for metrics in result.values():
        expect_true(abs(metrics.clustering - CLUSTERING_ONE) < TOLERANCE)
        expect_equal(metrics.triangles, TRIANGLES_ONE)
        expect_equal(metrics.core_number, CORE_NUMBER_TWO)


def test_all_structural_directed_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = nx.complete_graph(4, create_using=nx.DiGraph())
    result = compute_all_structural(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


# ===========================================================================
# Dataclass Frozen Tests
# ===========================================================================


def test_structural_metrics_frozen() -> None:
    """StructuralMetrics is frozen."""
    metrics = StructuralMetrics(
        clustering=0.5,
        triangles=2,
        core_number=3,
        constraint=0.25,
        effective_size=1.5,
    )
    assert_cannot_setattr(metrics, "clustering", 1.0)


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("n", "expected_core"),
    [
        (3, 2),  # K3 is 2-core
        (4, 3),  # K4 is 3-core
        (5, 4),  # K5 is 4-core
    ],
)
def test_complete_graph_core_numbers(n: int, expected_core: int) -> None:
    """Complete graphs Kn have core number n-1."""
    graph = nx.complete_graph(n)
    result = compute_core_number(graph)

    for core in result.values():
        expect_equal(core, expected_core)


@pytest.mark.parametrize(
    ("n", "expected_triangles_per_node"),
    [
        (3, 1),  # K3: 1 triangle
        (4, 3),  # K4: C(3,2) = 3
        (5, 6),  # K5: C(4,2) = 6
    ],
)
def test_complete_graph_triangles(n: int, expected_triangles_per_node: int) -> None:
    """Complete graphs have predictable triangle counts."""
    graph = nx.complete_graph(n)
    result = compute_triangles(graph)

    for count in result.values():
        expect_equal(count, expected_triangles_per_node)


@pytest.mark.parametrize(
    "spoke_count",
    [3, 5, 10],
)
def test_star_graph_clustering_zero(spoke_count: int) -> None:
    """Star graphs always have clustering zero."""
    graph = star_graph(spoke_count).to_undirected()
    result = compute_clustering_coefficient(graph)

    for clustering in result.values():
        expect_equal(clustering, CLUSTERING_ZERO)


@pytest.mark.parametrize(
    ("chain_length", "expected_core"),
    [
        (3, 1),
        (5, 1),
        (10, 1),
    ],
)
def test_chain_core_numbers(chain_length: int, expected_core: int) -> None:
    """Chain graphs always have core number 1."""
    graph = chain_graph(chain_length).to_undirected()
    result = compute_core_number(graph)

    for core in result.values():
        expect_equal(core, expected_core)
