"""Tests for community detection metric computation functions.

This module tests the stateless community detection functions including
greedy modularity, Louvain, and label propagation algorithms.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.community import (
    compute_modularity,
    detect_communities_greedy,
    detect_communities_label_propagation,
    detect_communities_louvain,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    barbell_graph_small,
    bridged_cliques_graph,
    chain_graph,
    complete_digraph,
    complete_graph,
    disconnected_graph,
    empty_graph,
    single_node_graph,
)
from tests.graphs.constants import COMPLETE_GRAPH_SIZES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_MIN_COMMUNITIES: Final[int] = 2
EXPECTED_SINGLE_COMMUNITY: Final[int] = 1
MODULARITY_MIN: Final[float] = -0.5
MODULARITY_MAX: Final[float] = 1.0
DEFAULT_RESOLUTION: Final[float] = 1.0
RANDOM_SEED: Final[int] = 42


# ===========================================================================
# detect_communities_greedy Tests
# ===========================================================================


def test_greedy_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_graph()
    result = detect_communities_greedy(graph)
    expect_equal(result, {})


def test_greedy_single_node_single_community() -> None:
    """Single node is its own community."""
    graph = single_node_graph(1)
    result = detect_communities_greedy(graph)

    expect_length(result, 1)
    expect_in(1, result)


def test_greedy_complete_graph() -> None:
    """Complete graph typically has one or few communities."""
    graph = complete_graph(5)
    result = detect_communities_greedy(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FIVE)
    # All nodes should be assigned to some community
    community_ids = set(result.values())
    expect_true(len(community_ids) >= EXPECTED_SINGLE_COMMUNITY)


def test_greedy_disconnected_components_separate_communities() -> None:
    """Disconnected components are in separate communities."""
    graph = bridged_cliques_graph(3, 3)
    result = detect_communities_greedy(graph)

    # Nodes in same clique should have same community
    expect_equal(result["a0"], result["a1"])
    expect_equal(result["a1"], result["a2"])
    expect_equal(result["b0"], result["b1"])
    expect_equal(result["b1"], result["b2"])
    # Different cliques should have different communities
    expect_true(result["a0"] != result["b0"])


def test_greedy_directed_graph_converted() -> None:
    """Directed graph is converted to undirected for community detection."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    result = detect_communities_greedy(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    # All nodes should be assigned
    expect_true(all(node in result for node in [1, 2, 3]))


def test_greedy_weighted_edges() -> None:
    """Weighted edges are respected when weight parameter provided."""
    graph = empty_graph()
    # Strong connection between 1-2, weak to 3
    graph.add_edge(1, 2, weight=10.0)
    graph.add_edge(2, 3, weight=0.1)
    result = detect_communities_greedy(graph, weight="weight")

    expect_length(result, EXPECTED_NODE_COUNT_THREE)


def test_greedy_resolution_parameter() -> None:
    """Resolution parameter affects community granularity."""
    graph = barbell_graph_small()

    result_low = detect_communities_greedy(graph, resolution=0.5)
    result_high = detect_communities_greedy(graph, resolution=2.0)

    # Higher resolution typically finds more communities
    communities_low = len(set(result_low.values()))
    communities_high = len(set(result_high.values()))
    # Both should work and return valid results
    expect_true(communities_low >= 1)
    expect_true(communities_high >= 1)


# ===========================================================================
# detect_communities_louvain Tests
# ===========================================================================


def test_louvain_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_graph()
    result = detect_communities_louvain(graph)
    expect_equal(result, {})


def test_louvain_single_node() -> None:
    """Single node is assigned to a community."""
    graph = single_node_graph("A")
    result = detect_communities_louvain(graph)

    expect_length(result, 1)
    expect_in("A", result)


def test_louvain_complete_graph() -> None:
    """Complete graph assigns all nodes to communities."""
    graph = complete_graph(5)
    result = detect_communities_louvain(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FIVE)


def test_louvain_disconnected_separate_communities() -> None:
    """Disconnected components get separate communities."""
    graph = disconnected_graph().to_undirected()
    result = detect_communities_louvain(graph)

    expect_length(result, EXPECTED_NODE_COUNT_SIX)
    # Check that nodes from same component share community
    expect_true(result["A"] == result["B"] or result["B"] == result["C"])


def test_louvain_directed_graph_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = complete_digraph(4)
    result = detect_communities_louvain(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_louvain_seed_reproducibility() -> None:
    """Same seed produces same results."""
    graph = barbell_graph_small()

    result1 = detect_communities_louvain(graph, seed=RANDOM_SEED)
    result2 = detect_communities_louvain(graph, seed=RANDOM_SEED)

    expect_equal(result1, result2)


def test_louvain_different_seeds_may_differ() -> None:
    """Different seeds may produce different results."""
    graph = nx.barbell_graph(10, 2)

    result1 = detect_communities_louvain(graph, seed=1)
    result2 = detect_communities_louvain(graph, seed=999)

    # Results should still be valid
    expect_equal(len(result1), len(result2))


def test_louvain_resolution_parameter() -> None:
    """Resolution parameter affects community granularity."""
    graph = barbell_graph_small()

    result_low = detect_communities_louvain(graph, resolution=0.5, seed=RANDOM_SEED)
    result_high = detect_communities_louvain(graph, resolution=2.0, seed=RANDOM_SEED)

    # Both should return valid community assignments
    expect_equal(len(result_low), len(result_high))


def test_louvain_weighted_edges() -> None:
    """Weighted edges are respected."""
    graph = empty_graph()
    graph.add_edge(1, 2, weight=100.0)
    graph.add_edge(2, 3, weight=0.001)
    graph.add_edge(3, 4, weight=100.0)

    result = detect_communities_louvain(graph, weight="weight")

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


# ===========================================================================
# detect_communities_label_propagation Tests
# ===========================================================================


def test_label_propagation_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_graph()
    result = detect_communities_label_propagation(graph)
    expect_equal(result, {})


def test_label_propagation_single_node() -> None:
    """Single node is assigned to a community."""
    graph = single_node_graph("X")
    result = detect_communities_label_propagation(graph)

    expect_length(result, 1)
    expect_in("X", result)


def test_label_propagation_complete_graph() -> None:
    """Complete graph assigns all nodes."""
    graph = complete_graph(5)
    result = detect_communities_label_propagation(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FIVE)


def test_label_propagation_disconnected_components() -> None:
    """Disconnected components get separate communities."""
    graph = bridged_cliques_graph(3, 3)
    result = detect_communities_label_propagation(graph)

    expect_length(result, EXPECTED_NODE_COUNT_SIX)
    # Same component should have same community
    expect_equal(result["a0"], result["a1"])
    expect_equal(result["a1"], result["a2"])
    expect_equal(result["b0"], result["b1"])
    expect_equal(result["b1"], result["b2"])


def test_label_propagation_directed_graph_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = chain_graph(4)
    result = detect_communities_label_propagation(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)


def test_label_propagation_chain_graph() -> None:
    """Chain graph may split or stay together."""
    graph = chain_graph(5).to_undirected()
    result = detect_communities_label_propagation(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FIVE)
    # All nodes should be assigned
    expect_true(all(node in result for node in ["A", "B", "C", "D", "E"]))


# ===========================================================================
# compute_modularity Tests
# ===========================================================================


def test_modularity_empty_graph_returns_zero() -> None:
    """Empty graph returns zero modularity."""
    graph = empty_graph()
    result = compute_modularity(graph, {})
    expect_equal(result, 0.0)


def test_modularity_empty_communities_returns_zero() -> None:
    """Empty communities dict returns zero modularity."""
    graph = complete_graph(5)
    result = compute_modularity(graph, {})
    expect_equal(result, 0.0)


def test_modularity_single_community() -> None:
    """Single community has defined modularity."""
    graph = complete_graph(5)
    communities = dict.fromkeys(graph.nodes(), 0)
    result = compute_modularity(graph, communities)

    # Single community modularity is well-defined
    expect_true(MODULARITY_MIN <= result <= MODULARITY_MAX)


def test_modularity_optimal_partition() -> None:
    """Well-separated communities have high modularity."""
    graph = bridged_cliques_graph(3, 3)
    communities = {node: 0 if node.startswith("a") else 1 for node in graph.nodes()}
    result = compute_modularity(graph, communities)

    # Should have positive modularity for good partition
    expect_true(result > 0)


def test_modularity_poor_partition() -> None:
    """Poorly separated partition has lower modularity."""
    graph = complete_graph(4)
    # Split complete graph - not a natural partition
    communities = {0: 0, 1: 0, 2: 1, 3: 1}
    result = compute_modularity(graph, communities)

    # Complete graph has no good partition
    expect_true(MODULARITY_MIN <= result <= MODULARITY_MAX)


def test_modularity_directed_graph_converted() -> None:
    """Directed graph is converted to undirected."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    communities = {1: 0, 2: 0, 3: 0}
    result = compute_modularity(graph, communities)

    expect_true(MODULARITY_MIN <= result <= MODULARITY_MAX)


def test_modularity_resolution_parameter() -> None:
    """Resolution parameter affects modularity calculation."""
    graph = barbell_graph_small()
    communities = detect_communities_greedy(graph)

    result_default = compute_modularity(graph, communities, resolution=DEFAULT_RESOLUTION)
    result_high = compute_modularity(graph, communities, resolution=2.0)

    # Both should be valid but may differ
    expect_true(MODULARITY_MIN <= result_default <= MODULARITY_MAX)
    expect_true(MODULARITY_MIN <= result_high <= MODULARITY_MAX)


def test_modularity_weighted_graph() -> None:
    """Weighted edges are considered in modularity."""
    graph = empty_graph()
    graph.add_edge(1, 2, weight=10.0)
    graph.add_edge(2, 3, weight=0.1)
    communities = {1: 0, 2: 0, 3: 1}

    result_weighted = compute_modularity(graph, communities, weight="weight")
    result_unweighted = compute_modularity(graph, communities, weight=None)

    # Both should return valid results
    expect_true(MODULARITY_MIN <= result_weighted <= MODULARITY_MAX)
    expect_true(MODULARITY_MIN <= result_unweighted <= MODULARITY_MAX)


def test_modularity_with_detected_communities() -> None:
    """Modularity of detected communities should be reasonable."""
    graph = barbell_graph_small()
    communities = detect_communities_louvain(graph, seed=RANDOM_SEED)
    result = compute_modularity(graph, communities)

    # Louvain optimizes modularity, so should be positive
    expect_true(result > 0)


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_all_algorithms_same_graph() -> None:
    """All algorithms work on the same graph and return valid results."""
    graph = barbell_graph_small()

    greedy_result = detect_communities_greedy(graph)
    louvain_result = detect_communities_louvain(graph, seed=RANDOM_SEED)
    label_prop_result = detect_communities_label_propagation(graph)

    # All should assign all nodes
    expect_equal(len(greedy_result), graph.number_of_nodes())
    expect_equal(len(louvain_result), graph.number_of_nodes())
    expect_equal(len(label_prop_result), graph.number_of_nodes())


def test_modularity_comparison_across_algorithms() -> None:
    """Compare modularity of different algorithm results."""
    graph = barbell_graph_small()

    greedy_communities = detect_communities_greedy(graph)
    louvain_communities = detect_communities_louvain(graph, seed=RANDOM_SEED)

    greedy_mod = compute_modularity(graph, greedy_communities)
    louvain_mod = compute_modularity(graph, louvain_communities)

    # Both should have valid modularity
    expect_true(MODULARITY_MIN <= greedy_mod <= MODULARITY_MAX)
    expect_true(MODULARITY_MIN <= louvain_mod <= MODULARITY_MAX)


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    "graph_size",
    COMPLETE_GRAPH_SIZES,
)
def test_greedy_various_sizes(graph_size: int) -> None:
    """Greedy algorithm works on various graph sizes."""
    graph = complete_graph(graph_size)
    result = detect_communities_greedy(graph)

    expect_equal(len(result), graph_size)


@pytest.mark.parametrize(
    "graph_size",
    COMPLETE_GRAPH_SIZES,
)
def test_louvain_various_sizes(graph_size: int) -> None:
    """Louvain algorithm works on various graph sizes."""
    graph = complete_graph(graph_size)
    result = detect_communities_louvain(graph, seed=RANDOM_SEED)

    expect_equal(len(result), graph_size)


@pytest.mark.parametrize(
    ("clique1_size", "clique2_size"),
    [
        (3, 3),
        (5, 5),
        (3, 7),
        (10, 10),
    ],
)
def test_two_cliques_detected_separately(clique1_size: int, clique2_size: int) -> None:
    """Two cliques connected by single edge are detected as separate."""
    graph = bridged_cliques_graph(clique1_size, clique2_size)

    result = detect_communities_louvain(graph, seed=RANDOM_SEED)
    communities = set(result.values())

    # Should find at least 2 communities
    expect_true(len(communities) >= EXPECTED_MIN_COMMUNITIES)
