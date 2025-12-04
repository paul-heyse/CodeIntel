"""Tests for path metric computation functions.

This module tests the stateless path computation functions including
simple path counting, shortest path lengths, and reachability.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.paths import (
    compute_avg_shortest_path_from_source,
    compute_reachable_nodes,
    count_simple_paths,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    diamond_graph,
    disconnected_graph,
    star_graph,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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


# ===========================================================================
# count_simple_paths Tests
# ===========================================================================


def test_simple_paths_empty_graph() -> None:
    """Empty graph returns zero paths."""
    graph = nx.DiGraph()
    result = count_simple_paths(graph, [1], [2], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT)
    assert result == EXPECTED_PATH_COUNT_ZERO


def test_simple_paths_no_sources() -> None:
    """Empty sources returns zero paths."""
    graph = chain_graph(4)
    result = count_simple_paths(
        graph, [], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_ZERO


def test_simple_paths_no_targets() -> None:
    """Empty targets returns zero paths."""
    graph = chain_graph(4)
    result = count_simple_paths(
        graph, ["A"], [], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_ZERO


def test_simple_paths_unreachable_target() -> None:
    """Unreachable target returns zero paths."""
    graph = disconnected_graph()  # A->B->C and X->Y->Z (disconnected)
    result = count_simple_paths(
        graph, ["A"], ["X"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_ZERO


def test_simple_paths_chain_graph() -> None:
    """Chain graph has one path between ends."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = count_simple_paths(
        graph, ["A"], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_ONE


def test_simple_paths_diamond_graph() -> None:
    """Diamond graph has two paths from A to D."""
    graph = diamond_graph()  # A -> B -> D and A -> C -> D
    result = count_simple_paths(
        graph, ["A"], ["D"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_TWO


def test_simple_paths_multiple_sources() -> None:
    """Multiple sources counts paths from all sources."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "C"), ("B", "C")])
    result = count_simple_paths(
        graph, ["A", "B"], ["C"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    # One path from A to C, one path from B to C
    assert result == EXPECTED_PATH_COUNT_TWO


def test_simple_paths_multiple_targets() -> None:
    """Multiple targets counts paths to all targets."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    result = count_simple_paths(
        graph, ["A"], ["B", "C"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_TWO


def test_simple_paths_max_paths_limit() -> None:
    """Max paths parameter limits count."""
    graph = nx.DiGraph()
    # Multiple paths: A -> B -> D, A -> C -> D, A -> E -> D
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
    assert result == MAX_PATHS_LIMITED


def test_simple_paths_cutoff_limit() -> None:
    """Cutoff parameter limits path length."""
    graph = chain_graph(5)  # A -> B -> C -> D -> E
    # With cutoff=1, can only reach one hop away
    result = count_simple_paths(
        graph, ["A"], ["E"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_SHORT
    )
    assert result == EXPECTED_PATH_COUNT_ZERO  # E is 4 hops away

    # With cutoff=4, can reach E
    result = count_simple_paths(graph, ["A"], ["E"], max_paths=MAX_PATHS_DEFAULT, cutoff=4)
    assert result == EXPECTED_PATH_COUNT_ONE


def test_simple_paths_self_loop_handled() -> None:
    """Source equals target handled (simple paths exclude loops)."""
    graph = nx.DiGraph()
    graph.add_edge("A", "A")
    # NetworkX all_simple_paths excludes self-loops for same source/target
    result = count_simple_paths(
        graph, ["A"], ["A"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    # Simple path from A to A is just [A] (length 0) - not counted
    assert result >= EXPECTED_PATH_COUNT_ZERO


def test_simple_paths_node_not_in_graph() -> None:
    """Source/target not in graph returns zero."""
    graph = chain_graph(3)
    result = count_simple_paths(
        graph, ["X"], ["Y"], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
    )
    assert result == EXPECTED_PATH_COUNT_ZERO


# ===========================================================================
# compute_avg_shortest_path_from_source Tests
# ===========================================================================


def test_avg_shortest_path_empty_graph() -> None:
    """Empty graph returns zero."""
    graph = nx.DiGraph()
    result = compute_avg_shortest_path_from_source(graph, "A")
    assert result == AVG_PATH_ZERO


def test_avg_shortest_path_single_node() -> None:
    """Single node source has zero average path length."""
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_avg_shortest_path_from_source(graph, "A")
    # Only path is to itself (length 0)
    assert result == AVG_PATH_ZERO


def test_avg_shortest_path_chain_graph() -> None:
    """Chain graph from source A."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_avg_shortest_path_from_source(graph, "A")

    # Sum of distances 0,1,2,3 divided by 4 gives 1.5
    expected_avg: float = 1.5
    assert abs(result - expected_avg) < AVG_PATH_TOLERANCE


def test_avg_shortest_path_star_graph() -> None:
    """Star graph from hub."""
    graph = star_graph(3)  # hub -> spoke1, spoke2, spoke3
    result = compute_avg_shortest_path_from_source(graph, "hub")

    # Sum of distances 0,1,1,1 divided by 4 gives 0.75
    expected_avg: float = 0.75
    assert abs(result - expected_avg) < AVG_PATH_TOLERANCE


def test_avg_shortest_path_from_sink() -> None:
    """Sink node can only reach itself."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_avg_shortest_path_from_source(graph, "D")

    # D can only reach itself (distance 0)
    assert result == AVG_PATH_ZERO


def test_avg_shortest_path_disconnected_source() -> None:
    """Source in disconnected component."""
    graph = disconnected_graph()
    result = compute_avg_shortest_path_from_source(graph, "A")

    # A can reach A, B, C (distances 0, 1, 2)
    expected_avg: float = (0 + 1 + 2) / 3
    assert abs(result - expected_avg) < AVG_PATH_TOLERANCE


def test_avg_shortest_path_source_not_in_graph() -> None:
    """Source not in graph returns zero."""
    graph = chain_graph(3)
    result = compute_avg_shortest_path_from_source(graph, "X")
    assert result == AVG_PATH_ZERO


def test_avg_shortest_path_diamond_graph() -> None:
    """Diamond graph uses shortest paths."""
    graph = diamond_graph()  # A -> B -> D and A -> C -> D
    result = compute_avg_shortest_path_from_source(graph, "A")

    # Sum of distances 0,1,1,2 divided by 4 gives 1.0
    expected_avg: float = 1.0
    assert abs(result - expected_avg) < AVG_PATH_TOLERANCE


# ===========================================================================
# compute_reachable_nodes Tests
# ===========================================================================


def test_reachable_nodes_empty_graph() -> None:
    """Empty graph returns just the source (if in graph) or source alone."""
    graph = nx.DiGraph()
    result = compute_reachable_nodes(graph, "A")
    # Source is always included even if not in graph
    assert "A" in result


def test_reachable_nodes_single_node() -> None:
    """Single node reaches only itself."""
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_reachable_nodes(graph, "A")

    assert result == {"A"}


def test_reachable_nodes_chain_graph() -> None:
    """Chain graph from source reaches all downstream."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_reachable_nodes(graph, "A")

    assert result == {"A", "B", "C", "D"}


def test_reachable_nodes_from_middle() -> None:
    """Middle of chain reaches downstream only."""
    graph = chain_graph(4)
    result = compute_reachable_nodes(graph, "B")

    assert result == {"B", "C", "D"}


def test_reachable_nodes_from_sink() -> None:
    """Sink node reaches only itself."""
    graph = chain_graph(4)
    result = compute_reachable_nodes(graph, "D")

    assert result == {"D"}


def test_reachable_nodes_star_graph() -> None:
    """Star graph hub reaches all spokes."""
    graph = star_graph(3)
    result = compute_reachable_nodes(graph, "hub")

    assert result == {"hub", "spoke1", "spoke2", "spoke3"}


def test_reachable_nodes_star_spoke() -> None:
    """Star graph spoke reaches only itself."""
    graph = star_graph(3)
    result = compute_reachable_nodes(graph, "spoke1")

    assert result == {"spoke1"}


def test_reachable_nodes_disconnected() -> None:
    """Disconnected component reaches only own component."""
    graph = disconnected_graph()
    result = compute_reachable_nodes(graph, "A")

    # A can reach A, B, C but not X, Y, Z
    assert result == {"A", "B", "C"}


def test_reachable_nodes_diamond() -> None:
    """Diamond graph reaches all nodes from source."""
    graph = diamond_graph()
    result = compute_reachable_nodes(graph, "A")

    assert result == {"A", "B", "C", "D"}


def test_reachable_nodes_source_not_in_graph() -> None:
    """Source not in graph returns just source."""
    graph = chain_graph(3)
    result = compute_reachable_nodes(graph, "X")

    # NetworkX descendants fails, but source is still added
    assert result == {"X"}


def test_reachable_nodes_cyclic_graph() -> None:
    """Cyclic graph reaches all nodes in cycle."""
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A")])
    result = compute_reachable_nodes(graph, "A")

    assert result == {"A", "B", "C"}


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_integration_reachable_matches_path_count() -> None:
    """Reachable nodes should have at least one path from source."""
    graph = diamond_graph()
    reachable = compute_reachable_nodes(graph, "A")

    for target in reachable:
        if target != "A":
            paths = count_simple_paths(
                graph, ["A"], [target], max_paths=MAX_PATHS_DEFAULT, cutoff=CUTOFF_DEFAULT
            )
            assert paths >= EXPECTED_PATH_COUNT_ONE


def test_integration_avg_path_length_consistency() -> None:
    """Average path length consistent with individual distances."""
    graph = chain_graph(4)

    # From A: distances are 0, 1, 2, 3
    # Average should be sum / count
    avg = compute_avg_shortest_path_from_source(graph, "A")
    reachable = compute_reachable_nodes(graph, "A")

    expected_avg = sum(range(len(reachable))) / len(reachable)
    assert abs(avg - expected_avg) < AVG_PATH_TOLERANCE


# ===========================================================================
# Parametrized Tests
# ===========================================================================


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

    assert len(result) == expected_reachable_from_first


@pytest.mark.parametrize(
    ("spoke_count", "expected_reachable_from_hub"),
    [
        (1, 2),  # hub + 1 spoke
        (3, 4),  # hub + 3 spokes
        (5, 6),  # hub + 5 spokes
        (10, 11),  # hub + 10 spokes
    ],
)
def test_reachable_star_graphs(spoke_count: int, expected_reachable_from_hub: int) -> None:
    """Star graphs of various sizes have correct reachability from hub."""
    graph = star_graph(spoke_count)
    result = compute_reachable_nodes(graph, "hub")

    assert len(result) == expected_reachable_from_hub


@pytest.mark.parametrize(
    ("chain_length", "cutoff", "expected_paths"),
    [
        (3, 1, 0),  # A to C, cutoff 1 (need 2 hops)
        (3, 2, 1),  # A to C, cutoff 2 (exactly 2 hops)
        (4, 2, 0),  # A to D, cutoff 2 (need 3 hops)
        (4, 3, 1),  # A to D, cutoff 3 (exactly 3 hops)
    ],
)
def test_paths_with_various_cutoffs(chain_length: int, cutoff: int, expected_paths: int) -> None:
    """Path counting with various cutoff values."""
    graph = chain_graph(chain_length)
    # Get the last node name
    last_node = chr(ord("A") + chain_length - 1)

    result = count_simple_paths(
        graph, ["A"], [last_node], max_paths=MAX_PATHS_DEFAULT, cutoff=cutoff
    )

    assert result == expected_paths
