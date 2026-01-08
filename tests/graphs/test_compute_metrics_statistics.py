"""Tests for graph statistics computation functions.

This module tests the stateless graph statistics functions including
degree computations, diameter estimation, and summary statistics.
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.build.graphs.compute.metrics.statistics import (
    GraphStatistics,
    compute_avg_shortest_path_length,
    compute_condensation_layer_count,
    compute_diameter_estimate,
    compute_graph_statistics,
    get_degree_values,
    get_degrees,
    get_in_degree_values,
    get_in_degrees,
    get_out_degree_values,
    get_out_degrees,
)
from codeintel.build.graphs.rx.algos import to_undirected_store
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)
from tests._helpers.fixtures.graphs import (
    chain_graph,
    complete_digraph,
    complete_graph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    empty_digraph,
    empty_graph,
    scc_with_tail_graph,
    single_node_digraph,
    star_graph,
    tree_graph,
)
from tests.graphs.constants import (
    CYCLE_SCC_SIZES,
    CYCLE_SIZE_SWEEP,
    SMALL_COMPLETE_GRAPH_SIZES,
    TREE_SHAPES,
)

EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_EDGE_COUNT_THREE: Final[int] = 3
EXPECTED_EDGE_COUNT_FOUR: Final[int] = 4
EXPECTED_EDGE_COUNT_TWELVE: Final[int] = 12
EXPECTED_LAYER_COUNT_ONE: Final[int] = 1
EXPECTED_LAYER_COUNT_THREE: Final[int] = 3
EXPECTED_LAYER_COUNT_FOUR: Final[int] = 4
EXPECTED_SCC_ONE: Final[int] = 1
EXPECTED_SCC_THREE: Final[int] = 3
EXPECTED_WCC_ONE: Final[int] = 1
EXPECTED_WCC_TWO: Final[int] = 2
EXPECTED_DEGREE_ONE: Final[int] = 1
EXPECTED_DEGREE_TWO: Final[int] = 2
EXPECTED_DEGREE_FOUR: Final[int] = 4
TOLERANCE: Final[float] = 0.01
DIAMETER_CHAIN_FIVE: Final[float] = 4.0
DIAMETER_COMPLETE: Final[float] = 1.0
DIAMETER_STAR: Final[float] = 2.0
AVG_PATH_CHAIN_FOUR: Final[float] = 5 / 3
AVG_PATH_COMPLETE: Final[float] = 1.0
DENSITY_COMPLETE_FOUR: Final[float] = 1.0
DENSITY_CHAIN_FOUR: Final[float] = 3 / 12


def test_in_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = empty_digraph()
    result = get_in_degrees(graph)
    expect_equal(result, [])


def test_in_degrees_single_node() -> None:
    """Single node has in-degree 0."""
    graph = single_node_digraph(1)
    result = get_in_degrees(graph)

    expect_length(result, 1)
    expect_equal(result[0], (1, 0))


def test_in_degrees_chain_graph() -> None:
    """Chain graph has correct in-degrees."""
    graph = chain_graph(4)
    result = get_in_degrees(graph)

    in_degree_dict = dict(result)
    expect_equal(in_degree_dict["A"], 0)
    expect_equal(in_degree_dict["B"], 1)
    expect_equal(in_degree_dict["C"], 1)
    expect_equal(in_degree_dict["D"], 1)


def test_in_degrees_star_graph() -> None:
    """Star graph hub has in-degree 0, spokes have 1."""
    graph = star_graph(3)
    result = get_in_degrees(graph)

    in_degree_dict = dict(result)
    expect_equal(in_degree_dict["hub"], 0)
    for i in range(1, 4):
        expect_equal(in_degree_dict[f"spoke{i}"], 1)


def test_in_degrees_diamond_graph() -> None:
    """Diamond graph D has in-degree 2."""
    graph = diamond_graph()
    result = get_in_degrees(graph)

    in_degree_dict = dict(result)
    expect_equal(in_degree_dict["A"], 0)
    expect_equal(in_degree_dict["B"], 1)
    expect_equal(in_degree_dict["C"], 1)
    expect_equal(in_degree_dict["D"], EXPECTED_DEGREE_TWO)


def test_out_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = empty_digraph()
    result = get_out_degrees(graph)
    expect_equal(result, [])


def test_out_degrees_chain_graph() -> None:
    """Chain graph has correct out-degrees."""
    graph = chain_graph(4)
    result = get_out_degrees(graph)

    out_degree_dict = dict(result)
    expect_equal(out_degree_dict["A"], 1)
    expect_equal(out_degree_dict["B"], 1)
    expect_equal(out_degree_dict["C"], 1)
    expect_equal(out_degree_dict["D"], 0)


def test_out_degrees_star_graph() -> None:
    """Star graph hub has out-degree equal to spoke count."""
    graph = star_graph(3)
    result = get_out_degrees(graph)

    out_degree_dict = dict(result)
    expect_equal(out_degree_dict["hub"], EXPECTED_NODE_COUNT_THREE)
    for i in range(1, 4):
        expect_equal(out_degree_dict[f"spoke{i}"], 0)


def test_out_degrees_diamond_graph() -> None:
    """Diamond graph A has out-degree 2."""
    graph = diamond_graph()
    result = get_out_degrees(graph)

    out_degree_dict = dict(result)
    expect_equal(out_degree_dict["A"], EXPECTED_DEGREE_TWO)
    expect_equal(out_degree_dict["B"], EXPECTED_DEGREE_ONE)
    expect_equal(out_degree_dict["C"], EXPECTED_DEGREE_ONE)
    expect_equal(out_degree_dict["D"], 0)


def test_degrees_empty_graph() -> None:
    """Empty graph returns empty list."""
    graph = empty_graph()
    result = get_degrees(graph)
    expect_equal(result, [])


def test_degrees_chain_undirected() -> None:
    """Undirected chain has correct degrees."""
    graph = to_undirected_store(chain_graph(4))
    result = get_degrees(graph)

    degree_dict = dict(result)
    expect_equal(degree_dict["A"], EXPECTED_DEGREE_ONE)
    expect_equal(degree_dict["B"], EXPECTED_DEGREE_TWO)
    expect_equal(degree_dict["C"], EXPECTED_DEGREE_TWO)
    expect_equal(degree_dict["D"], EXPECTED_DEGREE_ONE)


def test_degrees_complete_graph() -> None:
    """Complete graph has uniform degrees."""
    graph = complete_graph(5)
    result = get_degrees(graph)

    for _, degree in result:
        expect_equal(degree, EXPECTED_DEGREE_FOUR)


def test_in_degree_values() -> None:
    """Get just the in-degree values."""
    graph = chain_graph(3)
    result = get_in_degree_values(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    expect_in(0, result)
    expect_true(result.count(EXPECTED_DEGREE_ONE) >= EXPECTED_DEGREE_TWO)


def test_out_degree_values() -> None:
    """Get just the out-degree values."""
    graph = chain_graph(3)
    result = get_out_degree_values(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    expect_in(0, result)


def test_degree_values() -> None:
    """Get just the degree values (undirected)."""
    graph = to_undirected_store(chain_graph(3))
    result = get_degree_values(graph)

    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    expect_in(EXPECTED_DEGREE_ONE, result)
    expect_in(EXPECTED_DEGREE_TWO, result)


def test_diameter_empty_graph() -> None:
    """Empty graph returns None."""
    graph = empty_digraph()
    result = compute_diameter_estimate(graph)
    expect_true(result is None)


def test_diameter_single_node() -> None:
    """Single node has diameter 0."""
    graph = single_node_digraph("A")
    result = compute_diameter_estimate(graph)
    expect_equal(result, 0.0)


def test_diameter_chain_graph() -> None:
    """Chain graph diameter equals length."""
    graph = chain_graph(5)
    result = compute_diameter_estimate(graph)

    expect_equal(result, DIAMETER_CHAIN_FIVE)


def test_diameter_complete_graph() -> None:
    """Complete graph has diameter 1."""
    graph = complete_digraph(5)
    result = compute_diameter_estimate(graph)

    expect_equal(result, DIAMETER_COMPLETE)


def test_diameter_disconnected_uses_largest_component() -> None:
    """Disconnected graph uses largest component."""
    graph = disconnected_graph()
    result = compute_diameter_estimate(graph)

    expect_equal(result, DIAMETER_STAR)


def test_diameter_star_graph() -> None:
    """Star graph has diameter 2."""
    graph = star_graph(5)
    result = compute_diameter_estimate(graph)

    expect_equal(result, DIAMETER_STAR)


def test_avg_path_length_empty_graph() -> None:
    """Empty graph returns None."""
    graph = empty_digraph()
    result = compute_avg_shortest_path_length(graph)
    expect_true(result is None)


def test_avg_path_length_single_node() -> None:
    """Single node has avg path length 0."""
    graph = single_node_digraph("A")
    result = compute_avg_shortest_path_length(graph)
    expect_equal(result, 0.0)


def test_avg_path_length_chain_graph() -> None:
    """Chain graph has predictable avg path length."""
    graph = chain_graph(4)
    result = compute_avg_shortest_path_length(graph)

    expect_true(result is not None)
    if result is not None:
        expect_true(abs(result - AVG_PATH_CHAIN_FOUR) < TOLERANCE)


def test_avg_path_length_complete_graph() -> None:
    """Complete graph has avg path length 1."""
    graph = complete_digraph(5)
    result = compute_avg_shortest_path_length(graph)

    expect_true(result is not None)
    if result is not None:
        expect_true(abs(result - AVG_PATH_COMPLETE) < TOLERANCE)


def test_avg_path_length_disconnected_uses_largest() -> None:
    """Disconnected graph uses largest component."""
    graph = disconnected_graph()
    result = compute_avg_shortest_path_length(graph)

    expect_true(result is not None)


def test_condensation_layers_empty_graph() -> None:
    """Empty graph returns None."""
    graph = empty_digraph()
    result = compute_condensation_layer_count(graph)
    expect_true(result is None)


def test_condensation_layers_single_node() -> None:
    """Single node has 1 layer."""
    graph = single_node_digraph("A")
    result = compute_condensation_layer_count(graph)
    expect_equal(result, EXPECTED_LAYER_COUNT_ONE)


def test_condensation_layers_chain_graph() -> None:
    """Chain graph has N layers (each node is own SCC)."""
    graph = chain_graph(4)
    result = compute_condensation_layer_count(graph)
    expect_equal(result, EXPECTED_LAYER_COUNT_FOUR)


@pytest.mark.parametrize("cycle_size", CYCLE_SIZE_SWEEP)
def test_condensation_layers_cycle_graph(cycle_size: int) -> None:
    """Cycle graph has 1 layer (all nodes in one SCC)."""
    graph = cyclic_graph(cycle_size)
    result = compute_condensation_layer_count(graph)
    expect_equal(result, EXPECTED_LAYER_COUNT_ONE)


def test_condensation_layers_diamond_graph() -> None:
    """Diamond graph has 3 layers."""
    graph = diamond_graph()
    result = compute_condensation_layer_count(graph)

    expect_equal(result, EXPECTED_LAYER_COUNT_THREE)


def test_condensation_layers_mixed_graph() -> None:
    """Graph with SCC and DAG parts."""
    graph = scc_with_tail_graph()

    result = compute_condensation_layer_count(graph)

    expect_equal(result, EXPECTED_LAYER_COUNT_THREE)


@pytest.mark.parametrize(
    ("depth", "branching"),
    TREE_SHAPES,
)
def test_condensation_layers_tree_graphs(depth: int, branching: int) -> None:
    """Tree graphs have one layer per depth level."""
    graph = tree_graph(depth, branching)

    result = compute_condensation_layer_count(graph)

    expect_equal(result, depth + 1)


def test_statistics_empty_graph() -> None:
    """Empty graph returns zero statistics."""
    graph = empty_digraph()
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, 0)
    expect_equal(result.edge_count, 0)
    expect_equal(result.density, 0.0)
    expect_equal(result.avg_in_degree, 0.0)
    expect_equal(result.avg_out_degree, 0.0)
    expect_equal(result.strongly_connected_components, 0)
    expect_equal(result.weakly_connected_components, 0)
    expect_true(result.is_dag)


def test_statistics_single_node() -> None:
    """Single node graph statistics."""
    graph = single_node_digraph("A")
    with pytest.raises(ZeroDivisionError):
        compute_graph_statistics(graph)


def test_statistics_chain_graph() -> None:
    """Chain graph statistics."""
    graph = chain_graph(4)
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result.edge_count, EXPECTED_EDGE_COUNT_THREE)
    expect_true(abs(result.density - DENSITY_CHAIN_FOUR) < TOLERANCE)
    expect_equal(result.strongly_connected_components, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result.weakly_connected_components, EXPECTED_WCC_ONE)
    expect_true(result.is_dag)


def test_statistics_cyclic_graph() -> None:
    """Cyclic graph statistics."""
    graph = cyclic_graph(4)
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result.edge_count, EXPECTED_EDGE_COUNT_FOUR)
    expect_equal(result.strongly_connected_components, EXPECTED_SCC_ONE)
    expect_equal(result.weakly_connected_components, EXPECTED_WCC_ONE)
    expect_true(not result.is_dag)


def test_statistics_complete_graph() -> None:
    """Complete directed graph statistics."""
    graph = complete_digraph(4)
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result.edge_count, EXPECTED_EDGE_COUNT_TWELVE)
    expect_true(abs(result.density - DENSITY_COMPLETE_FOUR) < TOLERANCE)
    expect_equal(result.strongly_connected_components, EXPECTED_SCC_ONE)
    expect_true(not result.is_dag)


def test_statistics_disconnected_graph() -> None:
    """Disconnected graph statistics."""
    graph = disconnected_graph()
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, EXPECTED_NODE_COUNT_SIX)
    expect_equal(result.weakly_connected_components, EXPECTED_WCC_TWO)
    expect_true(result.is_dag)


def test_statistics_star_graph() -> None:
    """Star graph statistics."""
    graph = star_graph(3)
    result = compute_graph_statistics(graph)

    expect_equal(result.node_count, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result.edge_count, EXPECTED_EDGE_COUNT_THREE)
    expect_true(result.is_dag)


def test_statistics_returns_dataclass() -> None:
    """Returns GraphStatistics dataclass."""
    graph = chain_graph(3)
    result = compute_graph_statistics(graph)

    expect_true(isinstance(result, GraphStatistics))
    expect_true(hasattr(result, "node_count"))
    expect_true(hasattr(result, "edge_count"))
    expect_true(hasattr(result, "density"))
    expect_true(hasattr(result, "avg_in_degree"))
    expect_true(hasattr(result, "avg_out_degree"))
    expect_true(hasattr(result, "strongly_connected_components"))
    expect_true(hasattr(result, "weakly_connected_components"))
    expect_true(hasattr(result, "is_dag"))


def test_statistics_avg_degrees() -> None:
    """Average degree calculations."""
    graph = star_graph(4)
    result = compute_graph_statistics(graph)

    expect_true(abs(result.avg_out_degree - 0.8) < TOLERANCE)

    expect_true(abs(result.avg_in_degree - 0.8) < TOLERANCE)


def test_graph_statistics_frozen() -> None:
    """GraphStatistics is frozen."""
    stats = GraphStatistics(
        node_count=10,
        edge_count=20,
        density=0.5,
        avg_in_degree=2.0,
        avg_out_degree=2.0,
        strongly_connected_components=5,
        weakly_connected_components=1,
        is_dag=True,
    )
    assert_cannot_setattr(stats, "node_count", 100)


@pytest.mark.parametrize(
    ("node_count", "expected_scc_count"),
    [
        (2, 2),
        (5, 5),
        (10, 10),
    ],
)
def test_chain_scc_counts(node_count: int, expected_scc_count: int) -> None:
    """Chain graphs have one SCC per node."""
    graph = chain_graph(node_count)
    result = compute_graph_statistics(graph)

    expect_equal(result.strongly_connected_components, expected_scc_count)


@pytest.mark.parametrize(
    "cycle_size",
    CYCLE_SCC_SIZES,
)
def test_cycle_single_scc(cycle_size: int) -> None:
    """Cycle graphs have exactly one SCC."""
    graph = cyclic_graph(cycle_size)
    result = compute_graph_statistics(graph)

    expect_equal(result.strongly_connected_components, 1)


@pytest.mark.parametrize(
    "node_count",
    [3, 5],
)
def test_chain_is_dag(node_count: int) -> None:
    """Chain graphs are DAGs."""
    graph = chain_graph(node_count)
    result = compute_graph_statistics(graph)

    expect_true(result.is_dag)


@pytest.mark.parametrize(
    "n",
    SMALL_COMPLETE_GRAPH_SIZES,
)
def test_complete_edge_counts(n: int) -> None:
    """Complete directed graphs have n*(n-1) edges."""
    graph = complete_digraph(n)
    result = compute_graph_statistics(graph)

    expect_equal(result.edge_count, n * (n - 1))
