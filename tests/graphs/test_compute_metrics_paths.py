"""Tests for path metric computation functions.

This module tests the stateless path computation functions including
simple path counting, shortest path lengths, and reachability.
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.graphs.compute.metrics.paths import (
    compute_avg_shortest_path_from_source,
    compute_reachable_nodes,
    count_simple_paths,
)
from tests._helpers.assertions import expect_equal, expect_length, expect_true
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    empty_digraph,
    star_graph,
    tree_graph,
)
from tests.graphs.constants import STAR_SPOKE_SWEEP, TREE_SHAPES

EXPECTED_PATH_COUNT_ZERO: Final[int] = 0
EXPECTED_PATH_COUNT_ONE: Final[int] = 1
EXPECTED_PATH_COUNT_TWO: Final[int] = 2
EXPECTED_PATH_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
MAX_PATHS_DEFAULT: Final[int] = 100
MAX_PATHS_LIMITED: Final[int] = 2
CUTOFF_DEFAULT: Final[int] = 10
CUTOFF_SHORT: Final[int] = 1
AVG_PATH_TOLERANCE: Final[float] = 0.01
AVG_PATH_ZERO: Final[float] = 0.0


def test_simple_paths_empty_graph() -> None:
    """Empty graph returns zero paths."""
    graph = empty_digraph()
    result = count_simple_paths(graph, [1], [2], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT)
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)


def test_simple_paths_no_sources() -> None:
    """Empty sources returns zero paths."""
    graph = chain_graph(4)
    result = count_simple_paths(
        graph, [], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)


def test_simple_paths_no_targets() -> None:
    """Empty targets returns zero paths."""
    graph = chain_graph(4)
    result = count_simple_paths(
        graph, ["A"], [], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)


def test_simple_paths_unreachable_target() -> None:
    """Unreachable target returns zero paths."""
    graph = disconnected_graph()
    result = count_simple_paths(
        graph, ["A"], ["X"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)


def test_simple_paths_chain_graph() -> None:
    """Chain graph has one path between ends."""
    graph = chain_graph(4)
    result = count_simple_paths(
        graph, ["A"], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ONE)


def test_simple_paths_diamond_graph() -> None:
    """Diamond graph has two paths from A to D."""
    graph = diamond_graph()
    result = count_simple_paths(
        graph, ["A"], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_TWO)


def test_simple_paths_multiple_sources() -> None:
    """Multiple sources counts paths from all sources."""
    graph = star_graph(2, inward=True)
    result = count_simple_paths(
        graph, ["spoke1", "spoke2"], ["hub"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )

    expect_equal(result, EXPECTED_PATH_COUNT_TWO)


def test_simple_paths_multiple_targets() -> None:
    """Multiple targets counts paths to all targets."""
    graph = star_graph(2)
    result = count_simple_paths(
        graph, ["hub"], ["spoke1", "spoke2"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_TWO)


def test_simple_paths_max_paths_limit() -> None:
    """Max paths parameter limits count."""
    graph = empty_digraph()

    graph.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "E"),
            ("B", "D"),
            ("C", "D"),
            ("E", "D"),
        ]
    )
    result = count_simple_paths(
        graph, ["A"], ["D"], max_paths=MAX_PATHS_LIMITED, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, MAX_PATHS_LIMITED)


def test_simple_paths_cutoff_limit() -> None:
    """Cutoff parameter limits path length."""
    graph = chain_graph(5)

    result = count_simple_paths(
        graph, ["A"], ["E"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_SHORT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)

    result = count_simple_paths(graph, ["A"], ["E"], max_paths=MAX_PATHS_DEFAULT, cutoff=4)
    expect_equal(result, EXPECTED_PATH_COUNT_ONE)


def test_simple_paths_self_loop_handled() -> None:
    """Source equals target handled (simple paths exclude loops)."""
    graph = empty_digraph()
    graph.add_edge("A", "A")

    result = count_simple_paths(
        graph, ["A"], ["A"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )

    expect_true(result >= EXPECTED_PATH_COUNT_ZERO)


def test_simple_paths_node_not_in_graph() -> None:
    """Source/target not in graph returns zero."""
    graph = chain_graph(3)
    result = count_simple_paths(
        graph, ["X"], ["Y"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    expect_equal(result, EXPECTED_PATH_COUNT_ZERO)


def test_avg_shortest_path_empty_graph() -> None:
    """Empty graph returns zero."""
    graph = empty_digraph()
    result = compute_avg_shortest_path_from_source(graph, "A")
    expect_equal(result, AVG_PATH_ZERO)


def test_avg_shortest_path_single_node() -> None:
    """Single node source has zero average path length."""
    graph = chain_graph(1)
    result = compute_avg_shortest_path_from_source(graph, "A")

    expect_equal(result, AVG_PATH_ZERO)


def test_avg_shortest_path_chain_graph() -> None:
    """Chain graph from source A."""
    graph = chain_graph(4)
    result = compute_avg_shortest_path_from_source(graph, "A")

    expected_avg: float = 1.5
    expect_true(abs(result - expected_avg) < AVG_PATH_TOLERANCE)


def test_avg_shortest_path_star_graph() -> None:
    """Star graph from hub."""
    graph = star_graph(3)
    result = compute_avg_shortest_path_from_source(graph, "hub")

    expected_avg: float = 0.75
    expect_true(abs(result - expected_avg) < AVG_PATH_TOLERANCE)


def test_avg_shortest_path_from_sink() -> None:
    """Sink node can only reach itself."""
    graph = chain_graph(4)
    result = compute_avg_shortest_path_from_source(graph, "D")

    expect_equal(result, AVG_PATH_ZERO)


def test_avg_shortest_path_disconnected_source() -> None:
    """Source in disconnected component."""
    graph = disconnected_graph()
    result = compute_avg_shortest_path_from_source(graph, "A")

    expected_avg: float = (0 + 1 + 2) / 3
    expect_true(abs(result - expected_avg) < AVG_PATH_TOLERANCE)


def test_avg_shortest_path_source_not_in_graph() -> None:
    """Source not in graph returns zero."""
    graph = chain_graph(3)
    result = compute_avg_shortest_path_from_source(graph, "X")
    expect_equal(result, AVG_PATH_ZERO)


def test_avg_shortest_path_diamond_graph() -> None:
    """Diamond graph uses shortest paths."""
    graph = diamond_graph()
    result = compute_avg_shortest_path_from_source(graph, "A")

    expected_avg: float = 1.0
    expect_true(abs(result - expected_avg) < AVG_PATH_TOLERANCE)


def test_reachable_nodes_empty_graph() -> None:
    """Empty graph returns just the source (if in graph) or source alone."""
    graph = empty_digraph()
    result = compute_reachable_nodes(graph, "A")

    expect_true("A" in result)


def test_reachable_nodes_single_node() -> None:
    """Single node reaches only itself."""
    graph = chain_graph(1)
    result = compute_reachable_nodes(graph, "A")

    expect_equal(result, {"A"})


def test_reachable_nodes_chain_graph() -> None:
    """Chain graph from source reaches all downstream."""
    graph = chain_graph(4)
    result = compute_reachable_nodes(graph, "A")

    expect_equal(result, {"A", "B", "C", "D"})


def test_reachable_nodes_from_middle() -> None:
    """Middle of chain reaches downstream only."""
    graph = chain_graph(4)
    result = compute_reachable_nodes(graph, "B")

    expect_equal(result, {"B", "C", "D"})


def test_reachable_nodes_from_sink() -> None:
    """Sink node reaches only itself."""
    graph = chain_graph(4)
    result = compute_reachable_nodes(graph, "D")

    expect_equal(result, {"D"})


def test_reachable_nodes_star_graph() -> None:
    """Star graph hub reaches all spokes."""
    graph = star_graph(3)
    result = compute_reachable_nodes(graph, "hub")

    expect_equal(result, {"hub", "spoke1", "spoke2", "spoke3"})


def test_reachable_nodes_star_spoke() -> None:
    """Star graph spoke reaches only itself."""
    graph = star_graph(3)
    result = compute_reachable_nodes(graph, "spoke1")

    expect_equal(result, {"spoke1"})


def test_reachable_nodes_disconnected() -> None:
    """Disconnected component reaches only own component."""
    graph = disconnected_graph()
    result = compute_reachable_nodes(graph, "A")

    expect_equal(result, {"A", "B", "C"})


def test_reachable_nodes_diamond() -> None:
    """Diamond graph reaches all nodes from source."""
    graph = diamond_graph()
    result = compute_reachable_nodes(graph, "A")

    expect_equal(result, {"A", "B", "C", "D"})


def test_reachable_nodes_source_not_in_graph() -> None:
    """Source not in graph returns just source."""
    graph = chain_graph(3)
    result = compute_reachable_nodes(graph, "X")

    expect_equal(result, {"X"})


def test_reachable_nodes_cyclic_graph() -> None:
    """Cyclic graph reaches all nodes in cycle."""
    graph = cyclic_graph(3)
    result = compute_reachable_nodes(graph, "A")

    expect_equal(result, {"A", "B", "C"})


def test_integration_reachable_matches_path_count() -> None:
    """Reachable nodes should have at least one path from source."""
    graph = diamond_graph()
    reachable = compute_reachable_nodes(graph, "A")

    for target in reachable:
        if target != "A":
            paths = count_simple_paths(
                graph, ["A"], [target], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
            )
            expect_true(paths >= EXPECTED_PATH_COUNT_ONE)


def test_integration_avg_path_length_consistency() -> None:
    """Average path length consistent with individual distances."""
    graph = chain_graph(4)

    avg = compute_avg_shortest_path_from_source(graph, "A")
    reachable = compute_reachable_nodes(graph, "A")

    expected_avg = sum(range(len(reachable))) / len(reachable)
    expect_true(abs(avg - expected_avg) < AVG_PATH_TOLERANCE)


@pytest.mark.parametrize(
    ("chain_length", "expected_reachable_from_first"),
    [
        (2, 2),
        (3, 3),
        (5, 5),
        (10, 10),
    ],
)
def test_reachable_various_chains(chain_length: int, expected_reachable_from_first: int) -> None:
    """Chain graphs of various lengths have correct reachability."""
    graph = chain_graph(chain_length)
    result = compute_reachable_nodes(graph, "A")

    expect_length(result, expected_reachable_from_first)


@pytest.mark.parametrize(
    ("spoke_count", "expected_reachable_from_hub"),
    [(spokes, spokes + 1) for spokes in STAR_SPOKE_SWEEP],
)
def test_reachable_star_graphs(spoke_count: int, expected_reachable_from_hub: int) -> None:
    """Star graphs of various sizes have correct reachability from hub."""
    graph = star_graph(spoke_count)
    result = compute_reachable_nodes(graph, "hub")

    expect_length(result, expected_reachable_from_hub)


@pytest.mark.parametrize(
    ("depth", "branching"),
    TREE_SHAPES,
)
def test_reachable_tree_graphs(depth: int, branching: int) -> None:
    """Tree graphs reach all nodes from the root."""
    graph = tree_graph(depth, branching)
    reachable = compute_reachable_nodes(graph, "N0")

    expect_length(reachable, graph.number_of_nodes())


@pytest.mark.parametrize(
    ("chain_length", "cutoff", "expected_paths"),
    [
        (3, 1, 0),
        (3, 2, 1),
        (4, 2, 0),
        (4, 3, 1),
    ],
)
def test_paths_with_various_cutoffs(chain_length: int, cutoff: int, expected_paths: int) -> None:
    """Path counting with various cutoff values."""
    graph = chain_graph(chain_length)

    last_node = chr(ord("A") + chain_length - 1)

    result = count_simple_paths(
        graph, ["A"], [last_node], max_paths=MAX_PATHS_DEFAULT, cutoff=cutoff
    )

    expect_equal(result, expected_paths)
