"""Test graph statistics computation.

Test the pure computation functions for computing summary statistics
on directed graphs and the rustworkx store wrappers.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

from codeintel.build.graphs.compute.metrics.statistics import (
    GraphStatistics,
    compute_graph_statistics,
    get_degree_values,
    get_degrees,
    get_in_degree_values,
    get_in_degrees,
    get_out_degree_values,
    get_out_degrees,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from tests._helpers import assert_frozen
from tests._helpers.assertions import expect_equal, expect_false, expect_true

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


def _add_edges(store: RxGraphStore, edges: Iterable[tuple[Hashable, Hashable]]) -> None:
    for src, dst in edges:
        store.add_weighted_edge(src, dst, weight=1.0)


def _directed_store(edges: Iterable[tuple[Hashable, Hashable]]) -> RxGraphStore:
    store = RxGraphStore.directed()
    _add_edges(store, edges)
    return store


def _undirected_store(edges: Iterable[tuple[Hashable, Hashable]]) -> RxGraphStore:
    store = RxGraphStore.undirected()
    _add_edges(store, edges)
    return store


def _make_empty_graph() -> RxGraphStore:
    """
    Create an empty directed graph.

    Returns
    -------
    RxGraphStore
        An empty graph store.
    """
    return RxGraphStore.directed()


def _make_single_node() -> RxGraphStore:
    """
    Create a graph with single isolated node.

    Returns
    -------
    RxGraphStore
        A graph store with one node.
    """
    graph = RxGraphStore.directed()
    graph.ensure_node("A")
    return graph


def _make_simple_dag() -> RxGraphStore:
    """
    Create a simple DAG: A -> B -> C, A -> C.

    Returns
    -------
    RxGraphStore
        A simple directed acyclic graph store.
    """
    return _directed_store([("A", "B"), ("B", "C"), ("A", "C")])


def _make_simple_cycle() -> RxGraphStore:
    """
    Create a simple cycle: A -> B -> C -> A.

    Returns
    -------
    RxGraphStore
        A simple cycle graph store.
    """
    return _directed_store([("A", "B"), ("B", "C"), ("C", "A")])


def _make_call_graph() -> RxGraphStore:
    """
    Create a realistic call graph structure.

    Simulates function calls:
    - main calls init, process, cleanup
    - process calls validate, execute
    - execute calls helper1, helper2
    - validate also calls helper1 (shared utility)

    Returns
    -------
    RxGraphStore
        A realistic call graph store.
    """
    return _directed_store(
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


def _make_multiple_components() -> RxGraphStore:
    """
    Create a graph with multiple weakly connected components.

    Returns
    -------
    RxGraphStore
        A graph store with three disconnected components.
    """
    graph = RxGraphStore.directed()
    graph.add_weighted_edge("A", "B", weight=1.0)
    _add_edges(graph, [("X", "Y"), ("Y", "Z")])
    graph.ensure_node("Isolated")
    return graph


def _make_strongly_connected() -> RxGraphStore:
    """
    Create a strongly connected graph.

    Returns
    -------
    RxGraphStore
        A strongly connected directed graph store.
    """
    return _directed_store(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),
        ]
    )


def _make_complete_graph() -> RxGraphStore:
    """
    Create a complete directed graph (every node connects to every other).

    Returns
    -------
    RxGraphStore
        A complete directed graph store with 4 nodes.
    """
    nodes = ["A", "B", "C", "D"]
    edges = [(source, target) for source in nodes for target in nodes if source != target]
    return _directed_store(edges)


def _make_undirected_test() -> RxGraphStore:
    """
    Create an undirected graph for degree tests.

    Returns
    -------
    RxGraphStore
        An undirected test graph store.
    """
    return _undirected_store([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])


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
    expect_equal(stats.node_count, EXPECTED_NODES_10)
    expect_equal(stats.edge_count, EXPECTED_EDGES_15)
    expect_false(stats.is_dag)


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
    assert_frozen(stats, "node_count", 10)


def test_stats_empty_graph() -> None:
    """Empty graph produces zero statistics."""
    graph = _make_empty_graph()
    stats = compute_graph_statistics(graph)
    expect_equal(stats.node_count, 0)
    expect_equal(stats.edge_count, 0)
    expect_equal(stats.density, 0.0)
    expect_true(stats.is_dag)


def test_stats_single_node() -> None:
    """Single node graph statistics."""
    graph = _make_single_node()
    stats = compute_graph_statistics(graph)
    expect_equal(stats.node_count, 1)
    expect_equal(stats.edge_count, 0)
    expect_equal(stats.density, 0.0)
    expect_equal(stats.weakly_connected_components, 1)


def test_stats_simple_dag_is_dag() -> None:
    """Simple DAG is correctly identified."""
    graph = _make_simple_dag()
    stats = compute_graph_statistics(graph)
    expect_true(stats.is_dag)


def test_stats_cycle_is_not_dag() -> None:
    """Graph with cycle is not a DAG."""
    graph = _make_simple_cycle()
    stats = compute_graph_statistics(graph)
    expect_false(stats.is_dag)


def test_stats_node_and_edge_counts() -> None:
    """Correct node and edge counts."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)

    expect_equal(stats.node_count, EXPECTED_NODES_8)
    expect_equal(stats.edge_count, EXPECTED_EDGES_8)


def test_stats_density_calculation() -> None:
    """Density is calculated correctly."""
    graph = _make_complete_graph()
    stats = compute_graph_statistics(graph)

    expect_true(abs(stats.density - DENSITY_1_0) < TOLERANCE)


def test_stats_average_degrees() -> None:
    """Average in-degree equals average out-degree."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)

    expected_avg = stats.edge_count / stats.node_count
    expect_true(abs(stats.avg_in_degree - expected_avg) < TOLERANCE)
    expect_true(abs(stats.avg_out_degree - expected_avg) < TOLERANCE)


def test_stats_multiple_components() -> None:
    """Multiple components are counted correctly."""
    graph = _make_multiple_components()
    stats = compute_graph_statistics(graph)

    expect_equal(stats.weakly_connected_components, EXPECTED_WCC_3)

    expect_true(stats.strongly_connected_components >= EXPECTED_SCC_3)


def test_stats_strongly_connected_graph() -> None:
    """Strongly connected graph has 1 SCC."""
    graph = _make_strongly_connected()
    stats = compute_graph_statistics(graph)
    expect_equal(stats.strongly_connected_components, EXPECTED_SCC_1)


def test_stats_realistic_call_graph() -> None:
    """Statistics for realistic call graph."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)

    expect_true(stats.is_dag)

    expect_equal(stats.weakly_connected_components, EXPECTED_WCC_1)


def test_in_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.directed()
    result = get_in_degrees(graph)
    expect_equal(result, [])


def test_in_degrees_simple_chain() -> None:
    """In-degrees for simple chain."""
    graph = _directed_store([(1, 2), (2, 3)])
    result = get_in_degrees(graph)
    result_dict = dict(result)
    expect_equal(result_dict[1], 0)
    expect_equal(result_dict[2], 1)
    expect_equal(result_dict[3], 1)


def test_in_degrees_star_graph() -> None:
    """In-degrees for star graph (hub points to spokes)."""
    graph = _directed_store([(0, i) for i in range(1, 5)])
    result = get_in_degrees(graph)
    result_dict = dict(result)

    expect_equal(result_dict[0], 0)
    for i in range(1, 5):
        expect_equal(result_dict[i], 1)


def test_out_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.directed()
    result = get_out_degrees(graph)
    expect_equal(result, [])


def test_out_degrees_simple_chain() -> None:
    """Out-degrees for simple chain."""
    graph = _directed_store([(1, 2), (2, 3)])
    result = get_out_degrees(graph)
    result_dict = dict(result)
    expect_equal(result_dict[1], 1)
    expect_equal(result_dict[2], 1)
    expect_equal(result_dict[3], 0)


def test_out_degrees_star_graph() -> None:
    """Out-degrees for star graph."""
    graph = _directed_store([(0, i) for i in range(1, 5)])
    result = get_out_degrees(graph)
    result_dict = dict(result)

    expect_equal(result_dict[0], HUB_OUT_DEGREE)
    for i in range(1, 5):
        expect_equal(result_dict[i], 0)


def test_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.undirected()
    result = get_degrees(graph)
    expect_equal(result, [])


def test_degrees_simple_path() -> None:
    """Degrees for simple path."""
    graph = _undirected_store([(1, 2), (2, 3)])
    result = get_degrees(graph)
    result_dict = dict(result)
    expect_equal(result_dict[1], 1)
    expect_equal(result_dict[2], PATH_MIDDLE_DEGREE)
    expect_equal(result_dict[3], 1)


def test_degrees_triangle() -> None:
    """Degrees for triangle graph."""
    graph = _undirected_store([(1, 2), (2, 3), (3, 1)])
    result = get_degrees(graph)
    result_dict = dict(result)

    for node in [1, 2, 3]:
        expect_equal(result_dict[node], TRIANGLE_DEGREE)


def test_in_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.directed()
    result = get_in_degree_values(graph)
    expect_equal(result, [])


def test_in_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = _directed_store([(1, 2), (2, 3), (1, 3)])
    result = get_in_degree_values(graph)

    expect_equal(sorted(result), [0, 1, 2])


def test_out_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.directed()
    result = get_out_degree_values(graph)
    expect_equal(result, [])


def test_out_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = _directed_store([(1, 2), (1, 3), (2, 3)])
    result = get_out_degree_values(graph)

    expect_equal(sorted(result), [0, 1, 2])


def test_degree_values_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = RxGraphStore.undirected()
    result = get_degree_values(graph)
    expect_equal(result, [])


def test_degree_values_only() -> None:
    """Returns only values, not tuples."""
    graph = _make_undirected_test()
    result = get_degree_values(graph)

    expect_equal(sorted(result), [1, 2, 2, 3])


def test_statistics_consistency() -> None:
    """Statistics are internally consistent."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)

    expect_true(abs(stats.avg_in_degree - stats.avg_out_degree) < TOLERANCE)


def test_degree_functions_match_statistics() -> None:
    """Degree helper functions produce values consistent with statistics."""
    graph = _make_call_graph()
    stats = compute_graph_statistics(graph)
    in_values = get_in_degree_values(graph)
    out_values = get_out_degree_values(graph)

    expect_equal(sum(in_values), stats.edge_count)
    expect_equal(sum(out_values), stats.edge_count)


def test_various_graph_sizes() -> None:
    """Statistics work for various graph sizes."""
    for num_nodes in [5, 10, 20]:
        graph = _directed_store([(i, i + 1) for i in range(num_nodes - 1)])
        stats = compute_graph_statistics(graph)
        expect_equal(stats.node_count, num_nodes)
        expect_equal(stats.edge_count, num_nodes - 1)
