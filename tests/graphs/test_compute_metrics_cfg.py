"""Tests for CFG metric computation functions.

This module tests the stateless CFG computation functions including
dominator trees, dominance frontiers, loop headers, and path lengths.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.cfg import (
    DominanceMetrics,
    compute_all_dominance,
    compute_cfg_longest_path,
    compute_dominance_frontier,
    compute_dominator_depths,
    compute_dominator_tree,
    find_natural_loop_headers,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    cyclic_graph,
    diamond_graph,
    fork_join_cfg,
    while_loop_cfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_DEPTH_ZERO: Final[int] = 0
EXPECTED_DEPTH_ONE: Final[int] = 1
EXPECTED_DEPTH_TWO: Final[int] = 2
EXPECTED_DEPTH_THREE: Final[int] = 3
EXPECTED_PATH_LENGTH_TWO: Final[int] = 2
EXPECTED_PATH_LENGTH_THREE: Final[int] = 3


# ===========================================================================
# compute_dominator_tree Tests
# ===========================================================================


def test_dominator_tree_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_dominator_tree(graph, entry="A")
    expect_equal(result, {})


def test_dominator_tree_entry_not_in_graph_returns_empty() -> None:
    """Entry node not in graph returns empty dict."""
    graph = chain_graph(3)
    result = compute_dominator_tree(graph, entry="X")
    expect_equal(result, {})


def test_dominator_tree_single_node() -> None:
    """Single node graph with no edges returns empty dict.

    NetworkX immediate_dominators returns empty dict for isolated node.
    """
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_dominator_tree(graph, entry="A")
    expect_equal(result, {})


def test_dominator_tree_chain_graph() -> None:
    """Chain graph has linear dominator tree.

    Entry node is not included in result (only dominated nodes).
    """
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_dominator_tree(graph, entry="A")

    # Entry node 'A' is not in the result - only dominated nodes
    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    expect_false("A" in result)
    expect_equal(result["B"], "A")
    expect_equal(result["C"], "B")
    expect_equal(result["D"], "C")


def test_dominator_tree_diamond_graph() -> None:
    """Diamond graph has correct dominators.

    Entry node is not included in result (only dominated nodes).
    """
    graph = diamond_graph()  # A -> B, A -> C, B -> D, C -> D
    result = compute_dominator_tree(graph, entry="A")

    # Entry node 'A' is not in the result - only dominated nodes
    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    expect_false("A" in result)
    expect_equal(result["B"], "A")
    expect_equal(result["C"], "A")
    # D is dominated by A (the only common dominator of B and C)
    expect_equal(result["D"], "A")


def test_dominator_tree_multiple_paths() -> None:
    """Graph with multiple paths computes correct immediate dominators.

    Entry node is not included in result (only dominated nodes).
    """
    graph = diamond_graph()
    result = compute_dominator_tree(graph, entry="A")

    expect_false("A" in result)  # Entry node not in result
    expect_equal(result["B"], "A")
    expect_equal(result["C"], "A")
    expect_equal(result["D"], "A")  # A is the immediate dominator of D


# ===========================================================================
# compute_dominance_frontier Tests
# ===========================================================================


def test_dominance_frontier_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_dominance_frontier(graph, entry="A")
    expect_equal(result, {})


def test_dominance_frontier_entry_not_in_graph_returns_empty() -> None:
    """Entry node not in graph returns empty dict."""
    graph = nx.DiGraph([("A", "B")])
    result = compute_dominance_frontier(graph, entry="X")
    expect_equal(result, {})


def test_dominance_frontier_chain_graph_empty_frontiers() -> None:
    """Chain graph has empty dominance frontiers (no join points)."""
    graph = chain_graph(4)
    result = compute_dominance_frontier(graph, entry="A")

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    for node_frontier in result.values():
        expect_equal(node_frontier, frozenset())


def test_dominance_frontier_diamond_graph() -> None:
    """Diamond graph has D in frontier of B and C."""
    graph = diamond_graph()
    result = compute_dominance_frontier(graph, entry="A")

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    # B and C have D in their frontier (join point)
    expect_in("D", result["B"])
    expect_in("D", result["C"])
    # A dominates everything, has empty frontier
    expect_equal(result["A"], frozenset())


def test_dominance_frontier_if_then_else() -> None:
    """If-then-else pattern has correct frontier."""
    graph = fork_join_cfg(entry="entry", branch="if", left="then", right="else", join="join")
    result = compute_dominance_frontier(graph, entry="entry")

    # Then and else have join in their frontier
    expect_in("join", result["then"])
    expect_in("join", result["else"])


# ===========================================================================
# compute_dominator_depths Tests
# ===========================================================================


def test_dominator_depths_empty_returns_empty() -> None:
    """Empty idoms returns empty depths."""
    result = compute_dominator_depths({})
    expect_equal(result, {})


def test_dominator_depths_single_root() -> None:
    """Single root node has depth 0."""
    idoms: dict[str, str | None] = {"A": None}
    result = compute_dominator_depths(idoms)
    expect_equal(result, {"A": EXPECTED_DEPTH_ZERO})


def test_dominator_depths_chain() -> None:
    """Chain of dominators has incremental depths."""
    idoms: dict[str, str | None] = {
        "A": None,
        "B": "A",
        "C": "B",
        "D": "C",
    }
    result = compute_dominator_depths(idoms)

    expect_equal(result["A"], EXPECTED_DEPTH_ZERO)
    expect_equal(result["B"], EXPECTED_DEPTH_ONE)
    expect_equal(result["C"], EXPECTED_DEPTH_TWO)
    expect_equal(result["D"], EXPECTED_DEPTH_THREE)


def test_dominator_depths_tree() -> None:
    """Tree structure has correct depths."""
    # A dominates B and C, B dominates D
    idoms: dict[str, str | None] = {
        "A": None,
        "B": "A",
        "C": "A",
        "D": "B",
    }
    result = compute_dominator_depths(idoms)

    expect_equal(result["A"], EXPECTED_DEPTH_ZERO)
    expect_equal(result["B"], EXPECTED_DEPTH_ONE)
    expect_equal(result["C"], EXPECTED_DEPTH_ONE)
    expect_equal(result["D"], EXPECTED_DEPTH_TWO)


def test_dominator_depths_from_chain_graph() -> None:
    """Integration: depths from actual chain graph dominator tree."""
    graph = chain_graph(5)
    idoms = compute_dominator_tree(graph, entry="A")
    result = compute_dominator_depths(idoms)

    expect_equal(result["A"], EXPECTED_DEPTH_ZERO)
    expect_equal(result["B"], EXPECTED_DEPTH_ONE)
    expect_equal(result["C"], EXPECTED_DEPTH_TWO)
    expect_equal(result["D"], EXPECTED_DEPTH_THREE)


# ===========================================================================
# find_natural_loop_headers Tests
# ===========================================================================


def test_loop_headers_empty_graph_returns_empty() -> None:
    """Empty graph returns empty set."""
    graph = nx.DiGraph()
    result = find_natural_loop_headers(graph, entry="A")
    expect_equal(result, set())


def test_loop_headers_entry_not_in_graph_returns_empty() -> None:
    """Entry not in graph returns empty set."""
    graph = chain_graph(2)
    result = find_natural_loop_headers(graph, entry="X")
    expect_equal(result, set())


def test_loop_headers_dag_has_no_loops() -> None:
    """DAG has no loop headers."""
    graph = chain_graph(5)
    result = find_natural_loop_headers(graph, entry="A")
    expect_equal(result, set())


def test_loop_headers_simple_cycle() -> None:
    """Simple cycle has one loop header."""
    graph = cyclic_graph(3)  # A -> B -> C -> A
    result = find_natural_loop_headers(graph, entry="A")

    # A is the loop header (back edge from C to A)
    expect_in("A", result)


def test_loop_headers_self_loop() -> None:
    """Self-loop node is a loop header."""
    graph = nx.DiGraph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "B")  # Self-loop
    result = find_natural_loop_headers(graph, entry="A")

    expect_in("B", result)


def test_loop_headers_nested_loops() -> None:
    """Nested loops have multiple headers."""
    graph = nx.DiGraph()
    # Outer loop: A -> B -> C -> A
    # Inner loop: B -> D -> B
    graph.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("B", "D"),
            ("D", "B"),
        ]
    )
    result = find_natural_loop_headers(graph, entry="A")

    # A is outer loop header, B is inner loop header
    expect_in("A", result)
    expect_in("B", result)


def test_loop_headers_while_loop_pattern() -> None:
    """While loop pattern identifies correct header."""
    graph = while_loop_cfg()
    result = find_natural_loop_headers(graph, entry="entry")

    expect_in("condition", result)


# ===========================================================================
# compute_cfg_longest_path Tests
# ===========================================================================


def test_longest_path_empty_graph_returns_zero() -> None:
    """Empty graph returns 0."""
    graph = nx.DiGraph()
    result = compute_cfg_longest_path(graph)
    expect_equal(result, 0)


def test_longest_path_single_node_returns_zero() -> None:
    """Single node (no edges) returns 0."""
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_cfg_longest_path(graph)
    expect_equal(result, 0)


def test_longest_path_chain_graph() -> None:
    """Chain graph longest path equals number of edges."""
    graph = chain_graph(4)  # A -> B -> C -> D (3 edges)
    result = compute_cfg_longest_path(graph)
    expect_equal(result, EXPECTED_PATH_LENGTH_THREE)


def test_longest_path_diamond_graph() -> None:
    """Diamond graph longest path is 2 (A -> B -> D or A -> C -> D)."""
    graph = diamond_graph()
    result = compute_cfg_longest_path(graph)
    expect_equal(result, EXPECTED_PATH_LENGTH_TWO)


def test_longest_path_cyclic_graph_uses_condensation() -> None:
    """Cyclic graph computes longest path on condensation DAG."""
    graph = cyclic_graph(3)  # A -> B -> C -> A
    result = compute_cfg_longest_path(graph)

    # Condensation of a single SCC is a single node, so path length is 0
    expect_equal(result, 0)


def test_longest_path_mixed_dag_and_cycle() -> None:
    """Graph with both DAG part and cycle computes correct path."""
    graph = nx.DiGraph()
    # entry -> A -> B -> C -> A (cycle)
    #       -> exit
    graph.add_edges_from(
        [
            ("entry", "A"),
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),  # Back edge creating cycle
            ("C", "exit"),
        ]
    )
    result = compute_cfg_longest_path(graph)

    # Condensation has: entry -> SCC(A,B,C) -> exit
    # Path length is 2 edges
    expect_equal(result, EXPECTED_PATH_LENGTH_TWO)


# ===========================================================================
# compute_all_dominance Tests
# ===========================================================================


def test_all_dominance_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_all_dominance(graph, entry="A")
    expect_equal(result, {})


def test_all_dominance_returns_dataclass() -> None:
    """Returns DominanceMetrics dataclass for each node."""
    graph = chain_graph(3)
    result = compute_all_dominance(graph, entry="A")

    expect_length(result, EXPECTED_NODE_COUNT_THREE)
    for metrics in result.values():
        expect_is_instance(metrics, DominanceMetrics)
        expect_true(hasattr(metrics, "depth"))
        expect_true(hasattr(metrics, "frontier_size"))
        expect_true(hasattr(metrics, "is_loop_header"))


def test_all_dominance_chain_graph() -> None:
    """Chain graph has incremental depths and no loop headers."""
    graph = chain_graph(4)
    result = compute_all_dominance(graph, entry="A")

    expect_equal(result["A"].depth, EXPECTED_DEPTH_ZERO)
    expect_equal(result["B"].depth, EXPECTED_DEPTH_ONE)
    expect_equal(result["C"].depth, EXPECTED_DEPTH_TWO)
    expect_equal(result["D"].depth, EXPECTED_DEPTH_THREE)

    # No join points, so all frontiers are 0
    for metrics in result.values():
        expect_equal(metrics.frontier_size, 0)
        expect_false(metrics.is_loop_header)


def test_all_dominance_diamond_graph() -> None:
    """Diamond graph has correct frontiers."""
    graph = diamond_graph()
    result = compute_all_dominance(graph, entry="A")

    # B and C have D in frontier
    expect_equal(result["B"].frontier_size, 1)
    expect_equal(result["C"].frontier_size, 1)
    # A and D have empty frontiers
    expect_equal(result["A"].frontier_size, 0)
    expect_equal(result["D"].frontier_size, 0)


def test_all_dominance_cyclic_graph_identifies_headers() -> None:
    """Cyclic graph correctly identifies loop headers."""
    graph = cyclic_graph(3)
    result = compute_all_dominance(graph, entry="A")

    # A should be marked as loop header
    expect_true(result["A"].is_loop_header)


def test_all_dominance_complex_cfg() -> None:
    """Complex CFG with multiple features."""
    graph = nx.DiGraph()
    # entry -> if -> then -> join -> exit
    #            -> else -> join
    #       -> loop -> body -> loop (back edge)
    #               -> exit
    graph.add_edges_from(
        [
            ("entry", "if"),
            ("if", "then"),
            ("if", "else"),
            ("then", "join"),
            ("else", "join"),
            ("join", "loop"),
            ("loop", "body"),
            ("body", "loop"),  # Back edge
            ("loop", "exit"),
        ]
    )
    result = compute_all_dominance(graph, entry="entry")

    # Loop is a loop header due to back edge from body
    expect_true(result["loop"].is_loop_header)
    # If branches join at join node - then/else have join in frontier
    expect_true(result["then"].frontier_size >= 1)
    expect_true(result["else"].frontier_size >= 1)


# ===========================================================================
# Dataclass Frozen Tests
# ===========================================================================


def test_dominance_metrics_frozen() -> None:
    """DominanceMetrics is frozen."""
    metrics = DominanceMetrics(
        depth=1,
        frontier_size=2,
        is_loop_header=True,
    )
    assert_cannot_setattr(metrics, "depth", 5)


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("node_count", "expected_max_depth"),
    [
        (2, 1),
        (3, 2),
        (5, 4),
        (10, 9),
    ],
)
def test_chain_graph_depths_parametrized(node_count: int, expected_max_depth: int) -> None:
    """Chain graphs of various sizes have correct max depth."""
    graph = chain_graph(node_count)
    idoms = compute_dominator_tree(graph, entry="A")
    depths = compute_dominator_depths(idoms)

    max_depth = max(depths.values())
    expect_equal(max_depth, expected_max_depth)


@pytest.mark.parametrize(
    "cycle_size",
    [2, 3, 5, 10],
)
def test_cycle_graphs_have_loop_headers(cycle_size: int) -> None:
    """Cycle graphs of various sizes have loop headers."""
    graph = cyclic_graph(cycle_size)
    result = find_natural_loop_headers(graph, entry="A")

    expect_true(len(result) > 0)
