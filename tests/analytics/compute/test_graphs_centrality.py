"""Test graph centrality metric computation.

Test the pure computation functions for computing PageRank and
betweenness centrality on directed graphs using real NetworkX graphs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.analytics.compute.graphs.centrality import (
    CentralityMetrics,
    compute_betweenness,
    compute_pagerank,
)

# =============================================================================
# Constants
# =============================================================================

EXPECTED_NODES_4 = 4
EXPECTED_NODES_5 = 5
EXPECTED_NODES_7 = 7
EXPECTED_TOP_3 = 3
TOLERANCE = 0.001
DENSE_GRAPH_RANGE_TOLERANCE = 0.1
PAGERANK_SUM = 1.0

# Test data constants
TEST_PAGERANK = 0.25
TEST_BETWEENNESS = 0.5
TEST_IN_DEGREE = 3
TEST_OUT_DEGREE = 2

# =============================================================================
# Test Data: Realistic Graph Structures
# =============================================================================


def _make_simple_chain() -> nx.DiGraph:
    """
    Create a simple linear chain graph: A -> B -> C -> D.

    Returns
    -------
    nx.DiGraph
        A directed chain graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return graph


def _make_simple_cycle() -> nx.DiGraph:
    """
    Create a simple cycle graph: A -> B -> C -> A.

    Returns
    -------
    nx.DiGraph
        A directed cycle graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return graph


def _make_star_graph() -> nx.DiGraph:
    """
    Create a star graph with hub node pointing to spokes: Hub -> {A, B, C, D}.

    Returns
    -------
    nx.DiGraph
        An out-star directed graph.
    """
    graph = nx.DiGraph()
    for spoke in ["A", "B", "C", "D"]:
        graph.add_edge("Hub", spoke)
    return graph


def _make_reverse_star_graph() -> nx.DiGraph:
    """
    Create a reverse star with spokes pointing to hub: {A, B, C, D} -> Hub.

    Returns
    -------
    nx.DiGraph
        An in-star directed graph.
    """
    graph = nx.DiGraph()
    for spoke in ["A", "B", "C", "D"]:
        graph.add_edge(spoke, "Hub")
    return graph


def _make_call_graph_realistic() -> nx.DiGraph:
    """
    Create a realistic call graph structure.

    Simulates a typical codebase call hierarchy:
    - main() calls process_request() and handle_error()
    - process_request() calls validate(), execute(), log_result()
    - validate() and execute() share utility functions

    Returns
    -------
    nx.DiGraph
        A realistic call graph.
    """
    graph = nx.DiGraph()
    # Main entry point
    graph.add_edge("main", "process_request")
    graph.add_edge("main", "handle_error")
    # Request processing
    graph.add_edge("process_request", "validate")
    graph.add_edge("process_request", "execute")
    graph.add_edge("process_request", "log_result")
    # Validation
    graph.add_edge("validate", "check_input")
    graph.add_edge("validate", "sanitize")
    # Execution
    graph.add_edge("execute", "fetch_data")
    graph.add_edge("execute", "transform")
    graph.add_edge("execute", "save_result")
    # Shared utilities
    graph.add_edge("validate", "format_error")
    graph.add_edge("execute", "format_error")
    graph.add_edge("handle_error", "format_error")
    graph.add_edge("handle_error", "log_result")
    return graph


def _make_disconnected_components() -> nx.DiGraph:
    """
    Create a graph with multiple disconnected components.

    Returns
    -------
    nx.DiGraph
        A graph with three disconnected components.
    """
    graph = nx.DiGraph()
    # Component 1: A -> B -> C
    graph.add_edges_from([("A", "B"), ("B", "C")])
    # Component 2: X -> Y -> Z
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    # Component 3: isolated node
    graph.add_node("Isolated")
    return graph


def _make_dense_cluster() -> nx.DiGraph:
    """
    Create a densely connected cluster.

    Returns
    -------
    nx.DiGraph
        A complete directed graph with 5 nodes.
    """
    graph = nx.DiGraph()
    nodes = ["N1", "N2", "N3", "N4", "N5"]
    # Create edges from each node to all others (complete directed graph)
    for source in nodes:
        for target in nodes:
            if source != target:
                graph.add_edge(source, target)
    return graph


# =============================================================================
# CentralityMetrics Dataclass Tests
# =============================================================================


def test_metrics_create_all_fields() -> None:
    """Create metrics dataclass with all fields."""
    metrics = CentralityMetrics(
        node_id="test_node",
        pagerank=TEST_PAGERANK,
        betweenness=TEST_BETWEENNESS,
        in_degree=TEST_IN_DEGREE,
        out_degree=TEST_OUT_DEGREE,
    )
    assert metrics.node_id == "test_node"
    assert metrics.pagerank == TEST_PAGERANK
    assert metrics.betweenness == TEST_BETWEENNESS
    assert metrics.in_degree == TEST_IN_DEGREE
    assert metrics.out_degree == TEST_OUT_DEGREE


def test_metrics_is_frozen() -> None:
    """Metrics dataclass is immutable (frozen)."""
    metrics = CentralityMetrics(
        node_id="test",
        pagerank=0.1,
        betweenness=0.2,
        in_degree=1,
        out_degree=1,
    )
    with pytest.raises(AttributeError):
        metrics.pagerank = 0.5  # type: ignore[misc]


# =============================================================================
# compute_pagerank Tests
# =============================================================================


def test_pagerank_empty_graph() -> None:
    """Empty graph returns empty PageRank dictionary."""
    graph = nx.DiGraph()
    result = compute_pagerank(graph)
    assert result == {}


def test_pagerank_single_node() -> None:
    """Single node gets PageRank of 1.0."""
    graph = nx.DiGraph()
    graph.add_node("single")
    result = compute_pagerank(graph)
    assert "single" in result
    assert abs(result["single"] - PAGERANK_SUM) < TOLERANCE


def test_pagerank_simple_chain() -> None:
    """PageRank flows through chain graph."""
    graph = _make_simple_chain()
    result = compute_pagerank(graph)
    # All 4 nodes should be present
    assert len(result) == EXPECTED_NODES_4
    # D (end of chain) should have highest PageRank due to receiving flow
    assert result["D"] > result["A"]


def test_pagerank_cycle_equal() -> None:
    """Nodes in simple cycle have equal PageRank."""
    graph = _make_simple_cycle()
    result = compute_pagerank(graph)
    # All nodes in cycle should have similar PageRank
    values = list(result.values())
    assert max(values) - min(values) < TOLERANCE


def test_pagerank_star_hub_low() -> None:
    """Hub in out-star has lower PageRank (no incoming edges)."""
    graph = _make_star_graph()
    result = compute_pagerank(graph)
    # Hub has no incoming edges, spokes receive from hub
    # With damping, spokes should have higher PageRank
    hub_rank = result["Hub"]
    spoke_ranks = [result[s] for s in ["A", "B", "C", "D"]]
    avg_spoke = sum(spoke_ranks) / len(spoke_ranks)
    assert hub_rank < avg_spoke


def test_pagerank_reverse_star_hub_high() -> None:
    """Hub in in-star has higher PageRank (many incoming edges)."""
    graph = _make_reverse_star_graph()
    result = compute_pagerank(graph)
    hub_rank = result["Hub"]
    for spoke in ["A", "B", "C", "D"]:
        assert hub_rank > result[spoke]


def test_pagerank_realistic_call_graph() -> None:
    """PageRank identifies important functions in call graph."""
    graph = _make_call_graph_realistic()
    result = compute_pagerank(graph)
    # format_error is called by many functions, should be present
    assert "format_error" in result
    # log_result is called by multiple paths
    assert "log_result" in result


def test_pagerank_custom_alpha() -> None:
    """PageRank respects custom alpha (damping) parameter."""
    graph = _make_simple_chain()
    result_low = compute_pagerank(graph, alpha=0.5)
    result_high = compute_pagerank(graph, alpha=0.95)
    # Different alpha should give different distributions
    assert result_low != result_high


def test_pagerank_custom_max_iter() -> None:
    """PageRank respects custom max_iter parameter."""
    graph = _make_simple_chain()
    # Should converge quickly for simple graph
    result = compute_pagerank(graph, max_iter=10)
    assert len(result) == EXPECTED_NODES_4


def test_pagerank_custom_tolerance() -> None:
    """PageRank respects custom tolerance parameter."""
    graph = _make_simple_chain()
    result_low = compute_pagerank(graph, tol=1e-9)
    result_high = compute_pagerank(graph, tol=1e-3)
    # Both should produce results (may differ slightly in precision)
    assert len(result_low) == EXPECTED_NODES_4
    assert len(result_high) == EXPECTED_NODES_4


def test_pagerank_disconnected_components() -> None:
    """PageRank handles disconnected graph components."""
    graph = _make_disconnected_components()
    result = compute_pagerank(graph)
    # All 7 nodes should be present
    assert len(result) == EXPECTED_NODES_7
    # Isolated node should have non-zero PageRank (damping factor)
    assert "Isolated" in result
    assert result["Isolated"] > 0


def test_pagerank_sums_to_one() -> None:
    """PageRank values sum to approximately 1.0."""
    graph = _make_call_graph_realistic()
    result = compute_pagerank(graph)
    total = sum(result.values())
    assert abs(total - PAGERANK_SUM) < TOLERANCE


def test_pagerank_keys_are_strings() -> None:
    """PageRank result keys are always strings."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3)])  # Integer node IDs
    result = compute_pagerank(graph)
    for key in result:
        assert isinstance(key, str)


def test_pagerank_values_are_floats() -> None:
    """PageRank result values are always floats."""
    graph = _make_simple_chain()
    result = compute_pagerank(graph)
    for value in result.values():
        assert isinstance(value, float)


# =============================================================================
# compute_betweenness Tests
# =============================================================================


def test_betweenness_empty_graph() -> None:
    """Empty graph returns empty betweenness dictionary."""
    graph = nx.DiGraph()
    result = compute_betweenness(graph)
    assert result == {}


def test_betweenness_single_node_zero() -> None:
    """Single node has zero betweenness."""
    graph = nx.DiGraph()
    graph.add_node("single")
    result = compute_betweenness(graph)
    assert "single" in result
    assert result["single"] == 0.0


def test_betweenness_chain_middle_nodes_high() -> None:
    """Middle nodes in chain have higher betweenness."""
    graph = _make_simple_chain()  # A -> B -> C -> D
    result = compute_betweenness(graph)
    # B and C are on the shortest paths between A and D
    # End nodes A and D have lower betweenness
    assert result["B"] > result["A"]
    assert result["C"] > result["D"]


def test_betweenness_star_hub_high() -> None:
    """Hub in star graph has highest betweenness (all paths through it)."""
    graph = _make_star_graph()
    result = compute_betweenness(graph)
    hub_betweenness = result["Hub"]
    for spoke in ["A", "B", "C", "D"]:
        assert hub_betweenness >= result[spoke]


def test_betweenness_cycle_equal() -> None:
    """Nodes in simple cycle have equal betweenness."""
    graph = _make_simple_cycle()
    result = compute_betweenness(graph)
    values = list(result.values())
    assert max(values) - min(values) < TOLERANCE


def test_betweenness_normalized() -> None:
    """Normalized betweenness values are between 0 and 1."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph, normalized=True)
    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_betweenness_unnormalized() -> None:
    """Unnormalized betweenness can exceed 1."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph, normalized=False)
    # Just verify it runs and produces results
    assert len(result) == graph.number_of_nodes()


def test_betweenness_sampled_with_k() -> None:
    """Approximate betweenness with sample size k."""
    graph = _make_dense_cluster()
    # Sample 3 nodes for approximation
    result = compute_betweenness(graph, k=3)
    # Should still produce results for all nodes
    assert len(result) == EXPECTED_NODES_5


def test_betweenness_disconnected_components() -> None:
    """Betweenness handles disconnected components."""
    graph = _make_disconnected_components()
    result = compute_betweenness(graph)
    # All nodes should be present
    assert len(result) == EXPECTED_NODES_7
    # Isolated node has zero betweenness
    assert result["Isolated"] == 0.0


def test_betweenness_realistic_call_graph() -> None:
    """Betweenness identifies bridge functions in call graph."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph)
    # process_request is a bridge between main and many other functions
    assert "process_request" in result
    # process_request should have significant betweenness
    assert result["process_request"] > 0


def test_betweenness_keys_are_strings() -> None:
    """Betweenness result keys are always strings."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3)])
    result = compute_betweenness(graph)
    for key in result:
        assert isinstance(key, str)


def test_betweenness_values_are_floats() -> None:
    """Betweenness result values are always floats."""
    graph = _make_simple_chain()
    result = compute_betweenness(graph)
    for value in result.values():
        assert isinstance(value, float)


# =============================================================================
# Integration Tests
# =============================================================================


def test_both_metrics_same_nodes() -> None:
    """PageRank and betweenness produce results for same nodes."""
    graph = _make_call_graph_realistic()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)
    assert set(pagerank.keys()) == set(betweenness.keys())


def test_metrics_identify_different_importance() -> None:
    """PageRank and betweenness may rank nodes differently."""
    graph = _make_call_graph_realistic()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)
    # format_error has high PageRank (many incoming)
    # process_request has high betweenness (bridge node)
    # They measure different aspects of node importance
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    bc_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    # Top nodes may differ between metrics
    top_pr = [n for n, _ in pr_sorted[:EXPECTED_TOP_3]]
    top_bc = [n for n, _ in bc_sorted[:EXPECTED_TOP_3]]
    # Just verify both metrics produce reasonable rankings
    assert len(top_pr) == EXPECTED_TOP_3
    assert len(top_bc) == EXPECTED_TOP_3


def test_dense_graph_metrics() -> None:
    """Both metrics work on dense graphs."""
    graph = _make_dense_cluster()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)
    # All 5 nodes should have similar values in fully connected graph
    pr_values = list(pagerank.values())
    bc_values = list(betweenness.values())
    pr_range = max(pr_values) - min(pr_values)
    bc_range = max(bc_values) - min(bc_values)
    # Values should be relatively uniform in complete graph
    assert pr_range < DENSE_GRAPH_RANGE_TOLERANCE
    assert bc_range < DENSE_GRAPH_RANGE_TOLERANCE
