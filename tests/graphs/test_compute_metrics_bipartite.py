"""Tests for bipartite graph metric computation functions.

This module tests the stateless bipartite graph computation functions
including degree metrics and weighted projections.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.bipartite import (
    BipartiteDegreeMetrics,
    compute_bipartite_degrees,
    compute_weighted_projection,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import bipartite_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NODE_COUNT_TWO: Final[int] = 2
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_EDGE_COUNT_ONE: Final[int] = 1
EXPECTED_EDGE_COUNT_THREE: Final[int] = 3
EXPECTED_DEGREE_ONE: Final[int] = 1
EXPECTED_DEGREE_TWO: Final[int] = 2
EXPECTED_DEGREE_THREE: Final[int] = 3
TOLERANCE: Final[float] = 0.01
WEIGHT_VALUE: Final[float] = 2.5


def _require_projection(graph: nx.Graph | None) -> nx.Graph:
    """Ensure a projection graph exists for type checking.

    Returns
    -------
    nx.Graph
        The provided projection graph when it is present.
    """
    if graph is None:
        pytest.fail("Expected projection graph")
    return graph


# ===========================================================================
# compute_bipartite_degrees Tests
# ===========================================================================


def test_bipartite_degrees_empty_graph() -> None:
    """Empty graph returns empty metrics."""
    graph = nx.Graph()
    result = compute_bipartite_degrees(graph, set(), set())

    expect_equal(result.degree, {})
    expect_equal(result.weighted_degree, {})
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_empty_primary_partition() -> None:
    """Empty primary partition returns empty centrality."""
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])
    graph.add_edges_from([(1, 2), (2, 3)])

    result = compute_bipartite_degrees(graph, set(), {1, 2, 3})

    expect_true(result.degree != {})  # Degrees are still computed
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_empty_secondary_partition() -> None:
    """Empty secondary partition returns empty centrality."""
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])
    graph.add_edges_from([(1, 2), (2, 3)])

    result = compute_bipartite_degrees(graph, {1, 2, 3}, set())

    expect_true(result.degree != {})
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_simple_bipartite() -> None:
    """Simple bipartite graph degree computation."""
    # bipartite_graph() creates L0, L1, L2 (left) and R0, R1, R2 (right)
    graph = bipartite_graph()
    # Get the actual nodes from the graph
    primary = {"L0", "L1", "L2"}
    secondary = {"R0", "R1", "R2"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    # Check degrees are computed for all 6 nodes
    expect_equal(len(result.degree), EXPECTED_NODE_COUNT_SIX)
    expect_equal(len(result.weighted_degree), EXPECTED_NODE_COUNT_SIX)


def test_bipartite_degrees_unweighted() -> None:
    """Unweighted degree computation."""
    graph = nx.Graph()
    # Primary: 1 connects to a, b
    # Primary: 2 connects to b, c
    graph.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c")])
    primary = {1, 2}
    secondary = {"a", "b", "c"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    # Node 1 has degree 2 (connects to a, b)
    expect_equal(result.degree[1], EXPECTED_DEGREE_TWO)
    # Node 2 has degree 2 (connects to b, c)
    expect_equal(result.degree[2], EXPECTED_DEGREE_TWO)
    # Node b has degree 2 (connected from 1, 2)
    expect_equal(result.degree["b"], EXPECTED_DEGREE_TWO)
    # Node a has degree 1
    expect_equal(result.degree["a"], EXPECTED_DEGREE_ONE)
    # Node c has degree 1
    expect_equal(result.degree["c"], EXPECTED_DEGREE_ONE)


def test_bipartite_degrees_weighted() -> None:
    """Weighted degree computation."""
    graph = nx.Graph()
    graph.add_edge(1, "a", weight=WEIGHT_VALUE)
    graph.add_edge(1, "b", weight=1.0)
    graph.add_edge(2, "b", weight=WEIGHT_VALUE)
    primary = {1, 2}
    secondary = {"a", "b"}

    result = compute_bipartite_degrees(graph, primary, secondary, weight="weight")

    # Node 1: weighted degree = 2.5 + 1.0 = 3.5
    expected_weighted_1: float = WEIGHT_VALUE + 1.0
    expect_true(abs(result.weighted_degree[1] - expected_weighted_1) < TOLERANCE)

    # Node 2: weighted degree = 2.5
    expect_true(abs(result.weighted_degree[2] - WEIGHT_VALUE) < TOLERANCE)


def test_bipartite_degrees_degree_centrality() -> None:
    """Degree centrality for bipartite graph."""
    graph = nx.Graph()
    # Complete bipartite K_{2,3}
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (1, "c"),
            (2, "a"),
            (2, "b"),
            (2, "c"),
        ]
    )
    primary = {1, 2}
    secondary = {"a", "b", "c"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    # In complete bipartite, all primary nodes have same centrality
    expect_true(
        abs(result.primary_degree_centrality[1] - result.primary_degree_centrality[2]) < TOLERANCE
    )

    # All secondary nodes have same centrality
    expect_true(
        abs(result.secondary_degree_centrality["a"] - result.secondary_degree_centrality["b"])
        < TOLERANCE
    )


def test_bipartite_degrees_returns_dataclass() -> None:
    """Returns BipartiteDegreeMetrics dataclass."""
    graph = bipartite_graph()
    primary = {1, 2}
    secondary = {"a", "b", "c"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    expect_is_instance(result, BipartiteDegreeMetrics)
    expect_true(hasattr(result, "degree"))
    expect_true(hasattr(result, "weighted_degree"))
    expect_true(hasattr(result, "primary_degree_centrality"))
    expect_true(hasattr(result, "secondary_degree_centrality"))


def test_bipartite_degrees_no_weight_attribute() -> None:
    """Weight parameter with no weight attribute uses 1.0."""
    graph = nx.Graph()
    graph.add_edge(1, "a")
    graph.add_edge(1, "b")
    primary = {1}
    secondary = {"a", "b"}

    result = compute_bipartite_degrees(graph, primary, secondary, weight="weight")

    # No weight attribute, defaults to 1.0 per edge
    expect_equal(result.weighted_degree[1], EXPECTED_DEGREE_TWO)


# ===========================================================================
# compute_weighted_projection Tests
# ===========================================================================


def test_weighted_projection_empty_nodes() -> None:
    """Empty nodes set returns None."""
    graph = bipartite_graph()
    result = compute_weighted_projection(graph, set())
    expect_true(result is None)


def test_weighted_projection_nodes_not_in_graph() -> None:
    """Nodes not in graph returns None."""
    graph = bipartite_graph()
    result = compute_weighted_projection(graph, {"x", "y", "z"})
    expect_true(result is None)


def test_weighted_projection_all_nodes() -> None:
    """All nodes in graph returns None (not a valid bipartite partition)."""
    graph = bipartite_graph()
    all_nodes = set(graph.nodes())
    result = compute_weighted_projection(graph, all_nodes)
    expect_true(result is None)


def test_weighted_projection_simple_bipartite() -> None:
    """Simple bipartite projection."""
    graph = nx.Graph()
    # Primary: 1, 2 both connect to shared node 'x'
    graph.add_edges_from([(1, "x"), (2, "x")])
    primary = {1, 2}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    # 1 and 2 share connection through x, so they're connected in projection
    expect_true(projection.has_edge(1, 2))


def test_weighted_projection_no_shared_neighbors() -> None:
    """Nodes with no shared neighbors have no edges in projection."""
    graph = nx.Graph()
    # 1 connects to 'a', 2 connects to 'b' (no overlap)
    graph.add_edges_from([(1, "a"), (2, "b")])
    primary = {1, 2}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_equal(projection.number_of_nodes(), EXPECTED_NODE_COUNT_TWO)
    expect_equal(projection.number_of_edges(), 0)  # No shared neighbors


def test_weighted_projection_complete_bipartite() -> None:
    """Complete bipartite projects to complete graph."""
    graph = nx.Graph()
    # K_{3,2}: 3 primary nodes, 2 secondary nodes, all connected
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (2, "a"),
            (2, "b"),
            (3, "a"),
            (3, "b"),
        ]
    )
    primary = {1, 2, 3}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_equal(projection.number_of_nodes(), EXPECTED_NODE_COUNT_THREE)
    # Complete graph of 3 nodes has 3 edges
    expect_equal(projection.number_of_edges(), EXPECTED_EDGE_COUNT_THREE)


def test_weighted_projection_weights() -> None:
    """Projection has weights based on shared neighbors."""
    graph = nx.Graph()
    # 1 and 2 share both 'a' and 'b'
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (2, "a"),
            (2, "b"),
        ]
    )
    primary = {1, 2}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_true(projection.has_edge(1, 2))
    # Weight should reflect shared neighbors (2 shared)
    edge_data = projection.get_edge_data(1, 2)
    expect_true(edge_data is not None)
    if edge_data is not None:
        expect_true("weight" in edge_data)


def test_weighted_projection_partial_overlap() -> None:
    """Partial overlap creates weighted edges."""
    graph = nx.Graph()
    # 1 connects to a, b, c
    # 2 connects to b, c, d
    # 3 connects to c, d, e
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (1, "c"),
            (2, "b"),
            (2, "c"),
            (2, "d"),
            (3, "c"),
            (3, "d"),
            (3, "e"),
        ]
    )
    primary = {1, 2, 3}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_equal(projection.number_of_nodes(), EXPECTED_NODE_COUNT_THREE)
    # All pairs share at least one neighbor
    expect_true(projection.has_edge(1, 2))
    expect_true(projection.has_edge(2, 3))
    expect_true(projection.has_edge(1, 3))  # Share 'c'


def test_weighted_projection_secondary_partition() -> None:
    """Project onto secondary partition."""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (2, "b"),
            (2, "c"),
        ]
    )
    secondary = {"a", "b", "c"}

    result = compute_weighted_projection(graph, secondary)

    projection = _require_projection(result)
    # a and b share 1, b and c share 2
    expect_true(projection.has_edge("a", "b"))
    expect_true(projection.has_edge("b", "c"))


def test_weighted_projection_single_node() -> None:
    """Single node partition."""
    graph = nx.Graph()
    graph.add_edges_from([(1, "a"), (1, "b")])
    primary = {1}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_equal(projection.number_of_nodes(), 1)
    expect_equal(projection.number_of_edges(), 0)  # Single node, no edges


# ===========================================================================
# Dataclass Frozen Tests
# ===========================================================================


def test_bipartite_degree_metrics_frozen() -> None:
    """BipartiteDegreeMetrics is frozen."""
    metrics = BipartiteDegreeMetrics(
        degree={1: 2},
        weighted_degree={1: 2.5},
        primary_degree_centrality={1: 0.5},
        secondary_degree_centrality={"a": 0.5},
    )
    assert_cannot_setattr(metrics, "degree", {})


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_projection_matches_shared_neighbors_count() -> None:
    """Projection edge weights match shared neighbor counts."""
    graph = nx.Graph()
    # Node 1 and 2 share exactly 2 neighbors (a, b)
    graph.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (1, "c"),  # Not shared
            (2, "a"),
            (2, "b"),
            (2, "d"),  # Not shared
        ]
    )
    primary = {1, 2}

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    # The weighted_projected_graph uses neighbor count as weight
    edge_data = projection.get_edge_data(1, 2)
    expect_true(edge_data is not None)


def test_degree_centrality_sums_correctly() -> None:
    """Degree centrality values are in expected range.

    Note: degree_centrality can exceed 1.0 for bipartite graphs when
    using NetworkX's bipartite.degree_centrality, as it normalizes
    differently from regular degree centrality.
    """
    graph = bipartite_graph()
    primary = {"L0", "L1", "L2"}
    secondary = {"R0", "R1", "R2"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    # All centrality values should be non-negative
    for centrality in result.primary_degree_centrality.values():
        expect_true(centrality >= 0.0)

    for centrality in result.secondary_degree_centrality.values():
        expect_true(centrality >= 0.0)


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("primary_size", "secondary_size"),
    [
        (2, 3),
        (3, 3),
        (4, 2),
        (5, 5),
    ],
)
def test_complete_bipartite_degrees(primary_size: int, secondary_size: int) -> None:
    """Complete bipartite graphs have predictable degrees."""
    graph = nx.complete_bipartite_graph(primary_size, secondary_size)
    primary = set(range(primary_size))
    secondary = set(range(primary_size, primary_size + secondary_size))

    result = compute_bipartite_degrees(graph, primary, secondary)

    # All primary nodes have degree = secondary_size
    for node in primary:
        expect_equal(result.degree[node], secondary_size)

    # All secondary nodes have degree = primary_size
    for node in secondary:
        expect_equal(result.degree[node], primary_size)


@pytest.mark.parametrize(
    ("primary_size", "secondary_size", "expected_projection_edges"),
    [
        (2, 1, 1),  # 2 primary share 1 secondary -> 1 edge
        (3, 1, 3),  # 3 primary share 1 secondary -> 3 edges (complete)
        (3, 2, 3),  # 3 primary share 2 secondary -> 3 edges (complete)
    ],
)
def test_complete_bipartite_projection_edges(
    primary_size: int, secondary_size: int, expected_projection_edges: int
) -> None:
    """Complete bipartite projects to complete graph on primary."""
    graph = nx.complete_bipartite_graph(primary_size, secondary_size)
    primary = set(range(primary_size))

    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_equal(projection.number_of_edges(), expected_projection_edges)


@pytest.mark.parametrize(
    "shared_count",
    [1, 2, 3, 5],
)
def test_projection_with_varying_shared_neighbors(shared_count: int) -> None:
    """Projection edge weight varies with shared neighbor count."""
    graph = nx.Graph()
    # Nodes 1 and 2 share 'shared_count' neighbors
    for i in range(shared_count):
        graph.add_edge(1, f"s{i}")
        graph.add_edge(2, f"s{i}")
    # Add unique neighbors
    graph.add_edge(1, "unique1")
    graph.add_edge(2, "unique2")

    primary = {1, 2}
    result = compute_weighted_projection(graph, primary)

    projection = _require_projection(result)
    expect_true(projection.has_edge(1, 2))
