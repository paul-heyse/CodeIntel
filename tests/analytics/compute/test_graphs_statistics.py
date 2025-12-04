"""Test graph statistics computation.

Test the pure computation functions for computing summary statistics
on directed graphs and the type-safe NetworkX wrappers.
"""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.statistics import (
    GraphStatistics,
    compute_graph_statistics,
    get_degree_values,
    get_degrees,
    get_in_degree_values,
    get_in_degrees,
    get_out_degree_values,
    get_out_degrees,
)

# =============================================================================
# Constants
# =============================================================================

TOLERANCE = 0.001
EXPECTED_NODES_10 = 10
EXPECTED_EDGES_15 = 15
EXPECTED_NODES_5 = 5
EXPECTED_EDGES_5 = 5
EXPECTED_NODES_8 = 8
EXPECTED_EDGES_8 = 8
DENSITY_0_17 = 0.17
DENSITY_0_25 = 0.25
DENSITY_1_0 = 1.0
AVG_DEGREE_1_5 = 1.5
AVG_DEGREE_1_0 = 1.0
EXPECTED_SCC_1 = 1
EXPECTED_SCC_2 = 2
EXPECTED_SCC_3 = 3
EXPECTED_WCC_1 = 1
EXPECTED_WCC_3 = 3
HUB_OUT_DEGREE = 4
TRIANGLE_DEGREE = 2
PATH_MIDDLE_DEGREE = 2

# =============================================================================
# Test Data: Realistic Graph Structures
# =============================================================================


def _make_empty_graph() -> nx.DiGraph:
    """
    Create an empty directed graph.

    Returns
    -------
    nx.DiGraph
        An empty graph.
    """
    return nx.DiGraph()


def _make_single_node() -> nx.DiGraph:
    """
    Create a graph with single isolated node.

    Returns
    -------
    nx.DiGraph
        A graph with one node.
    """
    graph = nx.DiGraph()
    graph.add_node("A")
    return graph


def _make_simple_dag() -> nx.DiGraph:
    """
    Create a simple DAG: A -> B -> C, A -> C.

    Returns
    -------
    nx.DiGraph
        A simple directed acyclic graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return graph


def _make_simple_cycle() -> nx.DiGraph:
    """
    Create a simple cycle: A -> B -> C -> A.

    Returns
    -------
    nx.DiGraph
        A simple cycle graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return graph


def _make_call_graph() -> nx.DiGraph:
    """
    Create a realistic call graph structure.

    Simulates function calls:
    - main calls init, process, cleanup
    - process calls validate, execute
    - execute calls helper1, helper2
    - validate also calls helper1 (shared utility)

    Returns
    -------
    nx.DiGraph
        A realistic call graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("main", "init"),
            ("main", "process"),
            ("main", "cleanup"),
            ("process", "validate"),
            ("process", "execute"),
            ("validate", "helper1"),
            ("execute", "helper1"),
            ("execute", "helper2"),
        ]
    )
    return graph


def _make_multiple_components() -> nx.DiGraph:
    """
    Create a graph with multiple weakly connected components.

    Returns
    -------
    nx.DiGraph
        A graph with three disconnected components.
    """
    graph = nx.DiGraph()
    # Component 1: A -> B
    graph.add_edge("A", "B")
    # Component 2: X -> Y -> Z
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    # Component 3: isolated
    graph.add_node("Isolated")
    return graph


def _make_strongly_connected() -> nx.DiGraph:
    """
    Create a strongly connected graph.

    Returns
    -------
    nx.DiGraph
        A strongly connected directed graph.
    """
    graph = nx.DiGraph()
    # All nodes can reach each other
    graph.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),  # Back edge to form strong component
        ]
    )
    return graph


def _make_complete_graph() -> nx.DiGraph:
    """
    Create a complete directed graph (every node connects to every other).

    Returns
    -------
    nx.DiGraph
        A complete directed graph with 4 nodes.
    """
    graph = nx.DiGraph()
    nodes = ["A", "B", "C", "D"]
    for source in nodes:
        for target in nodes:
            if source != target:
                graph.add_edge(source, target)
    return graph


def _make_undirected_test() -> nx.Graph:
    """
    Create an undirected graph for degree tests.

    Returns
    -------
    nx.Graph
        An undirected test graph.
    """
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])
    return graph


# =============================================================================
# GraphStatistics Dataclass Tests
# =============================================================================


def test_statistics_create_all_fields() -> None:
    """Create statistics dataclass with all fields."""
    stats = GraphStatistics(
        node_count=EXPECTED_NODES_10,
        edge_count=EXPECTED_EDGES_15,
        density=DENSITY_0_17,
        avg_in_degree=AVG_DEGREE_1_5,
        avg_out_degree=AVG_DEGREE_1_5,
        strongly_connected_components=EXPECTED_SCC_2,
        weakly_connected_components=EXPECTED_WCC_1,
        is_dag=False,
    )
    assert stats.node_count == EXPECTED_NODES_10
    assert stats.edge_count == EXPECTED_EDGES_15
    assert not stats.is_dag


def test_statistics_is_frozen() -> None:
    """Statistics dataclass is immutable (frozen)."""
    stats = GraphStatistics(
        node_count=EXPECTED_NODES_5,
        edge_count=EXPECTED_EDGES_5,
        density=DENSITY_0_25,
        avg_in_degree=AVG_DEGREE_1_0,
        avg_out_degree=AVG_DEGREE_1_0,
        strongly_connected_components=EXPECTED_SCC_1,
        weakly_connected_components=EXPECTED_WCC_1,
        is_dag=True,
    )
    with pytest.raises(AttributeError):
        stats.node_count = 10  # type: ignore[misc]


# =============================================================================
# compute_graph_statistics Tests
# =============================================================================


def test_stats_empty_graph() -> None:
    """Empty graph produces zero statistics."""
    graph = _make_empty_graph()
    stats = compute_graph_statistics(graph)
    assert stats.node_count == 0
    assert stats.edge_count == 0
    assert stats.density == 0.0
    assert stats.is_dag


def test_stats_single_node() -> None:
    """Single node graph statistics."""
    graph = _make_single_node()
    stats = compute_graph_statistics(graph)
    assert stats.node_count == 1
    assert stats.edge_count == 0
    assert stats.density == 0.0
    assert stats.weakly_connected_components == 1


def test_stats_simple_dag_is_dag() -> None:
    """Simple DAG is correctly identified."""
    graph = _make_simple_dag()
    stats = compute_graph_statistics(graph)
    assert stats.is_dag


def test_stats_cycle_is_not_dag() -> None:
    """Graph with cycle is not a DAG."""
    graph = _make_simple_cycle()
    stats = compute_graph_statistics(graph)
    assert not stats.is_dag


def test_stats_node_and_edge_counts() -> None:
    """Correct node and edge counts."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    # main, init, process, cleanup, validate, execute, helper1, helper2
    assert stats.node_count == EXPECTED_NODES_8
    assert stats.edge_count == EXPECTED_EDGES_8


def test_stats_density_calculation() -> None:
    """Density is calculated correctly."""
    graph = _make_complete_graph()  # 4 nodes, 12 edges
    stats = compute_graph_statistics(graph)
    # For directed graph: density = edges / (nodes * (nodes - 1))
    # 12 / (4 * 3) = 1.0
    assert abs(stats.density - DENSITY_1_0) < TOLERANCE


def test_stats_average_degrees() -> None:
    """Average in-degree equals average out-degree."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    # In a directed graph, sum of in-degrees = sum of out-degrees = edge count
    # Therefore avg_in_degree = avg_out_degree = edges / nodes
    expected_avg = stats.edge_count / stats.node_count
    assert abs(stats.avg_in_degree - expected_avg) < TOLERANCE
    assert abs(stats.avg_out_degree - expected_avg) < TOLERANCE


def test_stats_multiple_components() -> None:
    """Multiple components are counted correctly."""
    graph = _make_multiple_components()
    stats = compute_graph_statistics(graph)
    # 3 weakly connected components
    assert stats.weakly_connected_components == EXPECTED_WCC_3
    # Each component is its own strongly connected component
    # A->B (2 SCCs), X->Y->Z (3 SCCs), Isolated (1 SCC) = 6 SCCs total
    assert stats.strongly_connected_components >= EXPECTED_SCC_3


def test_stats_strongly_connected_graph() -> None:
    """Strongly connected graph has 1 SCC."""
    graph = _make_strongly_connected()
    stats = compute_graph_statistics(graph)
    assert stats.strongly_connected_components == EXPECTED_SCC_1


def test_stats_realistic_call_graph() -> None:
    """Statistics for realistic call graph."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    # Call graph should be DAG (no recursion)
    assert stats.is_dag
    # Should have 1 weakly connected component
    assert stats.weakly_connected_components == EXPECTED_WCC_1


# =============================================================================
# nx_types Tests: get_in_degrees, get_out_degrees
# =============================================================================


def test_in_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
    result = get_in_degrees(graph)
    assert result == []


def test_in_degrees_simple_chain() -> None:
    """In-degrees for simple chain."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3)])
    result = get_in_degrees(graph)
    result_dict = dict(result)
    assert result_dict[1] == 0
    assert result_dict[2] == 1
    assert result_dict[3] == 1


def test_in_degrees_star_graph() -> None:
    """In-degrees for star graph (hub points to spokes)."""
    graph = nx.DiGraph()
    for i in range(1, 5):
        graph.add_edge(0, i)  # Hub (0) -> spokes (1-4)
    result = get_in_degrees(graph)
    result_dict = dict(result)
    # Hub has 0 in-degree, spokes have 1
    assert result_dict[0] == 0
    for i in range(1, 5):
        assert result_dict[i] == 1


def test_out_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
    result = get_out_degrees(graph)
    assert result == []


def test_out_degrees_simple_chain() -> None:
    """Out-degrees for simple chain."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3)])
    result = get_out_degrees(graph)
    result_dict = dict(result)
    assert result_dict[1] == 1
    assert result_dict[2] == 1
    assert result_dict[3] == 0


def test_out_degrees_star_graph() -> None:
    """Out-degrees for star graph."""
    graph = nx.DiGraph()
    for i in range(1, 5):
        graph.add_edge(0, i)
    result = get_out_degrees(graph)
    result_dict = dict(result)
    # Hub has out_degree=4, spokes have 0
    assert result_dict[0] == HUB_OUT_DEGREE
    for i in range(1, 5):
        assert result_dict[i] == 0


def test_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = get_degrees(graph)
    assert result == []


def test_degrees_simple_path() -> None:
    """Degrees for simple path."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3)])
    result = get_degrees(graph)
    result_dict = dict(result)
    assert result_dict[1] == 1
    assert result_dict[2] == PATH_MIDDLE_DEGREE
    assert result_dict[3] == 1


def test_degrees_triangle() -> None:
    """Degrees for triangle graph."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    result = get_degrees(graph)
    result_dict = dict(result)
    # All nodes should have degree 2 in triangle
    for node in [1, 2, 3]:
        assert result_dict[node] == TRIANGLE_DEGREE


# =============================================================================
# nx_types Tests: get_*_degree_values (values only)
# =============================================================================


def test_in_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
    result = get_in_degree_values(graph)
    assert result == []


def test_in_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
    result = get_in_degree_values(graph)
    # Node 1: in_degree=0, Node 2: in_degree=1, Node 3: in_degree=2
    assert sorted(result) == [0, 1, 2]


def test_out_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
    result = get_out_degree_values(graph)
    assert result == []


def test_out_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 3)])
    result = get_out_degree_values(graph)
    # Node 1: out_degree=2, Node 2: out_degree=1, Node 3: out_degree=0
    assert sorted(result) == [0, 1, 2]


def test_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = get_degree_values(graph)
    assert result == []


def test_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = _make_undirected_test()  # A-B, B-C, A-C, C-D
    result = get_degree_values(graph)
    # A: degree=2, B: degree=2, C: degree=3, D: degree=1
    assert sorted(result) == [1, 2, 2, 3]


# =============================================================================
# Integration Tests
# =============================================================================


def test_statistics_consistency() -> None:
    """Statistics are internally consistent."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    # avg_in_degree = avg_out_degree for any directed graph
    assert abs(stats.avg_in_degree - stats.avg_out_degree) < TOLERANCE


def test_degree_functions_match_statistics() -> None:
    """Degree helper functions produce values consistent with statistics."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    in_values = get_in_degree_values(graph)
    out_values = get_out_degree_values(graph)
    # Sum of in-degrees = sum of out-degrees = edge count
    assert sum(in_values) == stats.edge_count
    assert sum(out_values) == stats.edge_count


def test_various_graph_sizes() -> None:
    """Statistics work for various graph sizes."""
    for num_nodes in [5, 10, 20]:
        graph = nx.DiGraph()
        for i in range(num_nodes - 1):
            graph.add_edge(i, i + 1)
        stats = compute_graph_statistics(graph)
        assert stats.node_count == num_nodes
        assert stats.edge_count == num_nodes - 1
