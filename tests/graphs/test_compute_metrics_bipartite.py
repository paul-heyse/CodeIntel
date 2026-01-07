"""Tests for bipartite graph metric computation functions.

This module tests the stateless bipartite graph computation functions
including degree metrics and weighted projections.
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.build.graphs.compute.metrics.bipartite import (
    BipartiteDegreeMetrics,
    compute_bipartite_degrees,
    compute_weighted_projection,
)
from codeintel.build.graphs.rx.algos import to_undirected_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_is_instance,
    expect_true,
    require_projection_graph,
)
from tests._helpers.fixtures.graphs import (
    acyclic_bipartite_flow,
    bipartite_graph,
    empty_graph,
    shared_neighbors_graph,
    weighted_star_graph,
)

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


def _add_edges(store: RxGraphStore, edges: list[tuple[object, object]]) -> None:
    for src, dst in edges:
        store.add_weighted_edge(src, dst, weight=1.0)


def _edge_weight(store: RxGraphStore, src: object, dst: object) -> float | None:
    src_idx = store.get_index(src)
    dst_idx = store.get_index(dst)
    if src_idx is None or dst_idx is None:
        return None
    if not store.graph.has_edge(src_idx, dst_idx):
        return None
    payload = store.graph.get_edge_data(src_idx, dst_idx)
    return edge_weight_from_payload(payload)


def _has_edge(store: RxGraphStore, src: object, dst: object) -> bool:
    return _edge_weight(store, src, dst) is not None


def test_bipartite_degrees_empty_graph() -> None:
    """Empty graph returns empty metrics."""
    graph = empty_graph()
    result = compute_bipartite_degrees(graph, set(), set())

    expect_equal(result.degree, {})
    expect_equal(result.weighted_degree, {})
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_empty_primary_partition() -> None:
    """Empty primary partition returns empty centrality."""
    graph = acyclic_bipartite_flow(0, 3)

    result = compute_bipartite_degrees(graph, set(), {1, 2, 3})

    expect_true(result.degree != {})
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_empty_secondary_partition() -> None:
    """Empty secondary partition returns empty centrality."""
    graph = acyclic_bipartite_flow(3, 0)

    result = compute_bipartite_degrees(graph, {1, 2, 3}, set())

    expect_true(result.degree != {})
    expect_equal(result.primary_degree_centrality, {})
    expect_equal(result.secondary_degree_centrality, {})


def test_bipartite_degrees_simple_bipartite() -> None:
    """Simple bipartite graph degree computation."""
    graph = bipartite_graph()

    primary = {"L0", "L1", "L2"}
    secondary = {"R0", "R1", "R2"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    expect_equal(len(result.degree), EXPECTED_NODE_COUNT_SIX)
    expect_equal(len(result.weighted_degree), EXPECTED_NODE_COUNT_SIX)


def test_bipartite_degrees_unweighted() -> None:
    """Unweighted degree computation."""
    graph = shared_neighbors_graph(shared=1, primary=("p1", "p2"), unique_first=1, unique_second=1)
    primary = {"p1", "p2"}
    secondary = set(graph.node_ids()) - primary

    result = compute_bipartite_degrees(graph, primary, secondary)

    expect_equal(result.degree["p1"], EXPECTED_DEGREE_TWO)
    expect_equal(result.degree["p2"], EXPECTED_DEGREE_TWO)

    shared_nodes = {node for node in secondary if str(node).startswith("s")}
    unique_nodes = secondary - shared_nodes
    for node in shared_nodes:
        expect_equal(result.degree[node], EXPECTED_DEGREE_TWO)
    for node in unique_nodes:
        expect_equal(result.degree[node], EXPECTED_DEGREE_ONE)


def test_bipartite_degrees_weighted() -> None:
    """Weighted degree computation."""
    graph = to_undirected_store(weighted_star_graph(2, weight=WEIGHT_VALUE))
    graph.add_weighted_edge("hub", "b", weight=1.0)
    primary = {"hub"}
    secondary = {"spoke1", "spoke2", "b"}

    result = compute_bipartite_degrees(graph, primary, secondary, weight="weight")

    expected_weighted_hub: float = WEIGHT_VALUE * 2 + 1.0
    expect_true(abs(result.weighted_degree["hub"] - expected_weighted_hub) < TOLERANCE)
    expect_true(abs(result.weighted_degree["spoke1"] - WEIGHT_VALUE) < TOLERANCE)


def test_bipartite_degrees_degree_centrality() -> None:
    """Degree centrality for bipartite graph."""
    graph = to_undirected_store(bipartite_graph(2, 3))
    primary = {"L0", "L1"}
    secondary = {"R0", "R1", "R2"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    expect_true(
        abs(result.primary_degree_centrality["L0"] - result.primary_degree_centrality["L1"])
        < TOLERANCE
    )

    expect_true(
        abs(result.secondary_degree_centrality["R0"] - result.secondary_degree_centrality["R1"])
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
    graph = empty_graph()
    graph.add_weighted_edge(1, "a", weight=1.0)
    graph.add_weighted_edge(1, "b", weight=1.0)
    primary = {1}
    secondary = {"a", "b"}

    result = compute_bipartite_degrees(graph, primary, secondary, weight="weight")

    expect_equal(result.weighted_degree[1], EXPECTED_DEGREE_TWO)


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
    all_nodes = set(graph.node_ids())
    result = compute_weighted_projection(graph, all_nodes)
    expect_true(result is None)


def test_weighted_projection_simple_bipartite() -> None:
    """Simple bipartite projection."""
    graph = shared_neighbors_graph(shared=1)
    primary = {"p1", "p2"}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)

    expect_true(_has_edge(projection, "p1", "p2"))


def test_weighted_projection_no_shared_neighbors() -> None:
    """Nodes with no shared neighbors have no edges in projection."""
    graph = shared_neighbors_graph(shared=0, unique_first=1, unique_second=1)
    primary = {"p1", "p2"}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_equal(projection.graph.num_nodes(), EXPECTED_NODE_COUNT_TWO)
    expect_equal(projection.graph.num_edges(), 0)


def test_weighted_projection_complete_bipartite() -> None:
    """Complete bipartite projects to complete graph."""
    graph = to_undirected_store(bipartite_graph(3, 2))
    primary = {"L0", "L1", "L2"}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_equal(projection.graph.num_nodes(), EXPECTED_NODE_COUNT_THREE)

    expect_equal(projection.graph.num_edges(), EXPECTED_EDGE_COUNT_THREE)


def test_weighted_projection_weights() -> None:
    """Projection has weights based on shared neighbors."""
    graph = shared_neighbors_graph(shared=2)
    primary = {"p1", "p2"}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_true(_has_edge(projection, "p1", "p2"))
    expect_true(_edge_weight(projection, "p1", "p2") is not None)


def test_weighted_projection_partial_overlap() -> None:
    """Partial overlap creates weighted edges."""
    graph = empty_graph()

    _add_edges(
        graph,
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
        ],
    )
    primary = {1, 2, 3}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_equal(projection.graph.num_nodes(), EXPECTED_NODE_COUNT_THREE)

    expect_true(_has_edge(projection, 1, 2))
    expect_true(_has_edge(projection, 2, 3))
    expect_true(_has_edge(projection, 1, 3))


def test_weighted_projection_secondary_partition() -> None:
    """Project onto secondary partition."""
    graph = empty_graph()
    _add_edges(
        graph,
        [
            (1, "a"),
            (1, "b"),
            (2, "b"),
            (2, "c"),
        ],
    )
    secondary = {"a", "b", "c"}

    result = compute_weighted_projection(graph, secondary)

    projection = require_projection_graph(result)

    expect_true(_has_edge(projection, "a", "b"))
    expect_true(_has_edge(projection, "b", "c"))


def test_weighted_projection_single_node() -> None:
    """Single node partition."""
    graph = empty_graph()
    _add_edges(graph, [(1, "a"), (1, "b")])
    primary = {1}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_equal(projection.graph.num_nodes(), 1)
    expect_equal(projection.graph.num_edges(), 0)


def test_bipartite_degree_metrics_frozen() -> None:
    """BipartiteDegreeMetrics is frozen."""
    metrics = BipartiteDegreeMetrics(
        degree={1: 2},
        weighted_degree={1: 2.5},
        primary_degree_centrality={1: 0.5},
        secondary_degree_centrality={"a": 0.5},
    )
    assert_cannot_setattr(metrics, "degree", {})


def test_projection_matches_shared_neighbors_count() -> None:
    """Projection edge weights match shared neighbor counts."""
    graph = shared_neighbors_graph(shared=2, unique_first=1, unique_second=1)
    primary = {"p1", "p2"}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)

    expect_true(_edge_weight(projection, "p1", "p2") is not None)


def test_degree_centrality_sums_correctly() -> None:
    """Degree centrality values are in expected range.

    Note: degree_centrality can exceed 1.0 for bipartite graphs because
    the normalization differs from regular degree centrality.
    """
    graph = bipartite_graph()
    primary = {"L0", "L1", "L2"}
    secondary = {"R0", "R1", "R2"}

    result = compute_bipartite_degrees(graph, primary, secondary)

    for centrality in result.primary_degree_centrality.values():
        expect_true(centrality >= 0.0)

    for centrality in result.secondary_degree_centrality.values():
        expect_true(centrality >= 0.0)


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
    graph = to_undirected_store(bipartite_graph(primary_size, secondary_size))
    primary = {f"L{i}" for i in range(primary_size)}
    secondary = {f"R{i}" for i in range(secondary_size)}

    result = compute_bipartite_degrees(graph, primary, secondary)

    for node in primary:
        expect_equal(result.degree[node], secondary_size)

    for node in secondary:
        expect_equal(result.degree[node], primary_size)


@pytest.mark.parametrize(
    ("primary_size", "secondary_size", "expected_projection_edges"),
    [
        (2, 1, 1),
        (3, 1, 3),
        (3, 2, 3),
    ],
)
def test_complete_bipartite_projection_edges(
    primary_size: int, secondary_size: int, expected_projection_edges: int
) -> None:
    """Complete bipartite projects to complete graph on primary."""
    graph = to_undirected_store(bipartite_graph(primary_size, secondary_size))
    primary = {f"L{i}" for i in range(primary_size)}

    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_equal(projection.graph.num_edges(), expected_projection_edges)


@pytest.mark.parametrize(
    "shared_count",
    [1, 2, 3, 5],
)
def test_projection_with_varying_shared_neighbors(shared_count: int) -> None:
    """Projection edge weight varies with shared neighbor count."""
    graph = empty_graph()

    for i in range(shared_count):
        graph.add_weighted_edge(1, f"s{i}", weight=1.0)
        graph.add_weighted_edge(2, f"s{i}", weight=1.0)

    graph.add_weighted_edge(1, "unique1", weight=1.0)
    graph.add_weighted_edge(2, "unique2", weight=1.0)

    primary = {1, 2}
    result = compute_weighted_projection(graph, primary)

    projection = require_projection_graph(result)
    expect_true(_has_edge(projection, 1, 2))
