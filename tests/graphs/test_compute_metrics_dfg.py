"""Tests for DFG metric computation functions.

This module tests the stateless data flow graph computation functions
including path lengths, component analysis, def-use chains, and cycles.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.dfg import (
    DFGPathStats,
    compute_def_use_chains,
    compute_dfg_components,
    compute_dfg_density,
    compute_dfg_path_lengths,
    compute_use_def_chains,
    find_dfg_cycles,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    complete_digraph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    star_graph,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NODE_COUNT_THREE: Final[int] = 3
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_SINGLE_COMPONENT: Final[int] = 1
EXPECTED_TWO_COMPONENTS: Final[int] = 2
EXPECTED_MAX_DISTANCE_ONE: Final[int] = 1
EXPECTED_MAX_DISTANCE_TWO: Final[int] = 2
EXPECTED_MAX_DISTANCE_THREE: Final[int] = 3
EXPECTED_REACH_COUNT_ZERO: Final[int] = 0
EXPECTED_REACH_COUNT_ONE: Final[int] = 1
EXPECTED_REACH_COUNT_TWO: Final[int] = 2
EXPECTED_REACH_COUNT_THREE: Final[int] = 3
DENSITY_ZERO: Final[float] = 0.0
DENSITY_TOLERANCE: Final[float] = 0.01
MAX_DEPTH_DEFAULT: Final[int] = 100
MAX_DEPTH_LIMITED: Final[int] = 2
CYCLE_LIMIT_DEFAULT: Final[int] = 100
CYCLE_LIMIT_ONE: Final[int] = 1


# ===========================================================================
# compute_dfg_path_lengths Tests
# ===========================================================================


def test_path_lengths_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_dfg_path_lengths(graph)
    expect_equal(result, {})


def test_path_lengths_single_node() -> None:
    """Single node has zero distances."""
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_dfg_path_lengths(graph)

    expect_length(result, 1)
    expect_equal(result["A"].max_def_use_distance, 0)
    expect_equal(result["A"].avg_def_use_distance, 0.0)
    expect_equal(result["A"].reach_count, EXPECTED_REACH_COUNT_ZERO)


def test_path_lengths_chain_graph() -> None:
    """Chain graph has incremental distances."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_dfg_path_lengths(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)

    # A reaches B, C, D at distances 1, 2, 3
    expect_equal(result["A"].max_def_use_distance, EXPECTED_MAX_DISTANCE_THREE)
    expect_equal(result["A"].reach_count, EXPECTED_REACH_COUNT_THREE)

    # B reaches C, D at distances 1, 2
    expect_equal(result["B"].max_def_use_distance, EXPECTED_MAX_DISTANCE_TWO)
    expect_equal(result["B"].reach_count, EXPECTED_REACH_COUNT_TWO)

    # D reaches nothing (sink)
    expect_equal(result["D"].max_def_use_distance, 0)
    expect_equal(result["D"].reach_count, EXPECTED_REACH_COUNT_ZERO)


def test_path_lengths_diamond_graph() -> None:
    """Diamond graph has correct path lengths through multiple routes."""
    graph = diamond_graph()  # A -> B, A -> C, B -> D, C -> D
    result = compute_dfg_path_lengths(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)

    # A reaches B, C (dist 1) and D (dist 2)
    expect_equal(result["A"].max_def_use_distance, EXPECTED_MAX_DISTANCE_TWO)
    expect_equal(result["A"].reach_count, EXPECTED_REACH_COUNT_THREE)

    # B reaches only D
    expect_equal(result["B"].max_def_use_distance, EXPECTED_MAX_DISTANCE_ONE)
    expect_equal(result["B"].reach_count, EXPECTED_REACH_COUNT_ONE)


def test_path_lengths_star_graph() -> None:
    """Star graph has hub reaching all spokes at distance 1."""
    graph = star_graph(3)  # hub -> spoke1, spoke2, spoke3
    result = compute_dfg_path_lengths(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result["hub"].max_def_use_distance, EXPECTED_MAX_DISTANCE_ONE)
    expect_equal(result["hub"].reach_count, EXPECTED_REACH_COUNT_THREE)


def test_path_lengths_max_depth_limiting() -> None:
    """Max depth parameter limits search depth.

    With max_depth=2, we explore from nodes at depth 2 and can still
    find their successors at depth 3 (the check is `dist > max_depth`).
    """
    graph = chain_graph(10)  # A -> B -> ... -> J (long chain)
    result = compute_dfg_path_lengths(graph, max_depth=MAX_DEPTH_LIMITED)

    # A reaches B (1), C (2), and D (3) since we explore from depth 2
    expect_equal(result["A"].max_def_use_distance, EXPECTED_MAX_DISTANCE_THREE)
    expect_equal(result["A"].reach_count, EXPECTED_REACH_COUNT_THREE)


def test_path_lengths_avg_calculation() -> None:
    """Average distance is correctly calculated."""
    graph = nx.DiGraph()
    # A reaches B (dist 1), C (dist 2), D (dist 3)
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    result = compute_dfg_path_lengths(graph)

    # Avg for A: (1 + 2 + 3) / 3 = 2.0
    expected_avg: float = 2.0
    expect_true(abs(result["A"].avg_def_use_distance - expected_avg) < DENSITY_TOLERANCE)


def test_path_lengths_returns_dataclass() -> None:
    """Returns DFGPathStats dataclass for each node."""
    graph = chain_graph(3)
    result = compute_dfg_path_lengths(graph)

    for stats in result.values():
        expect_is_instance(stats, DFGPathStats)
        expect_true(hasattr(stats, "max_def_use_distance"))
        expect_true(hasattr(stats, "avg_def_use_distance"))
        expect_true(hasattr(stats, "reach_count"))


# ===========================================================================
# compute_dfg_components Tests
# ===========================================================================


def test_dfg_components_empty_graph_returns_empty() -> None:
    """Empty graph returns empty component lists."""
    graph = nx.DiGraph()
    scc, wcc = compute_dfg_components(graph)
    expect_equal(scc, [])
    expect_equal(wcc, [])


def test_dfg_components_single_node() -> None:
    """Single node is one SCC and one WCC."""
    graph = nx.DiGraph()
    graph.add_node("A")
    scc, wcc = compute_dfg_components(graph)

    expect_length(scc, EXPECTED_SINGLE_COMPONENT)
    expect_length(wcc, EXPECTED_SINGLE_COMPONENT)


def test_dfg_components_chain_graph() -> None:
    """Chain graph has N SCCs (trivial) and 1 WCC."""
    graph = chain_graph(4)
    scc, wcc = compute_dfg_components(graph)

    # Each node is its own SCC (no cycles)
    expect_length(scc, EXPECTED_NODE_COUNT_FOUR)
    # All connected weakly
    expect_length(wcc, EXPECTED_SINGLE_COMPONENT)


def test_dfg_components_cyclic_graph() -> None:
    """Cyclic graph has one non-trivial SCC."""
    graph = cyclic_graph(3)  # A -> B -> C -> A
    scc, wcc = compute_dfg_components(graph)

    # One SCC containing all nodes
    expect_length(scc, EXPECTED_SINGLE_COMPONENT)
    expect_equal(scc[0].size, EXPECTED_NODE_COUNT_THREE)
    # One WCC
    expect_length(wcc, EXPECTED_SINGLE_COMPONENT)


def test_dfg_components_disconnected_graph() -> None:
    """Disconnected graph has multiple WCCs."""
    graph = disconnected_graph()
    scc, wcc = compute_dfg_components(graph)

    # 6 nodes, all trivial SCCs (chains)
    expect_length(scc, EXPECTED_NODE_COUNT_SIX)
    # 2 disconnected components
    expect_length(wcc, EXPECTED_TWO_COMPONENTS)


def test_dfg_components_mixed_graph() -> None:
    """Graph with both cycle and dag parts."""
    graph = nx.DiGraph()
    # Cycle: A -> B -> C -> A
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # DAG part: D -> E
    graph.add_edges_from([("D", "E")])
    # Connection: C -> D
    graph.add_edge("C", "D")

    scc, wcc = compute_dfg_components(graph)

    # SCCs: {A,B,C}, {D}, {E}
    expect_length(scc, EXPECTED_NODE_COUNT_THREE)
    # All weakly connected
    expect_length(wcc, EXPECTED_SINGLE_COMPONENT)


# ===========================================================================
# compute_def_use_chains Tests
# ===========================================================================


def test_def_use_chains_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_def_use_chains(graph)
    expect_equal(result, {})


def test_def_use_chains_single_node() -> None:
    """Single node has empty chain."""
    graph = nx.DiGraph()
    graph.add_node("def")
    result = compute_def_use_chains(graph)

    expect_length(result, 1)
    expect_equal(result["def"], [])


def test_def_use_chains_chain_graph() -> None:
    """Chain graph has single use per def."""
    graph = chain_graph(4)  # A -> B -> C -> D
    result = compute_def_use_chains(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result["A"], ["B"])
    expect_equal(result["B"], ["C"])
    expect_equal(result["C"], ["D"])
    expect_equal(result["D"], [])


def test_def_use_chains_star_graph() -> None:
    """Star graph hub has multiple uses."""
    graph = star_graph(3)
    result = compute_def_use_chains(graph)

    # Hub has 3 uses (spokes)
    expect_length(result["hub"], EXPECTED_REACH_COUNT_THREE)
    expect_equal(set(result["hub"]), {"spoke1", "spoke2", "spoke3"})

    # Spokes have no uses
    expect_equal(result["spoke1"], [])
    expect_equal(result["spoke2"], [])
    expect_equal(result["spoke3"], [])


def test_def_use_chains_diamond_graph() -> None:
    """Diamond graph has correct def-use chains."""
    graph = diamond_graph()
    result = compute_def_use_chains(graph)

    # A defines to B and C
    expect_equal(set(result["A"]), {"B", "C"})
    # B and C both use D
    expect_equal(result["B"], ["D"])
    expect_equal(result["C"], ["D"])
    # D is a sink
    expect_equal(result["D"], [])


# ===========================================================================
# compute_use_def_chains Tests
# ===========================================================================


def test_use_def_chains_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_use_def_chains(graph)
    expect_equal(result, {})


def test_use_def_chains_single_node() -> None:
    """Single node has empty chain."""
    graph = nx.DiGraph()
    graph.add_node("use")
    result = compute_use_def_chains(graph)

    expect_length(result, 1)
    expect_equal(result["use"], [])


def test_use_def_chains_chain_graph() -> None:
    """Chain graph has single def per use."""
    graph = chain_graph(4)
    result = compute_use_def_chains(graph)

    expect_length(result, EXPECTED_NODE_COUNT_FOUR)
    expect_equal(result["A"], [])
    expect_equal(result["B"], ["A"])
    expect_equal(result["C"], ["B"])
    expect_equal(result["D"], ["C"])


def test_use_def_chains_diamond_graph() -> None:
    """Diamond graph D has multiple definitions."""
    graph = diamond_graph()
    result = compute_use_def_chains(graph)

    # D has two definitions (B and C)
    expect_equal(set(result["D"]), {"B", "C"})
    # A has no definitions
    expect_equal(result["A"], [])


def test_use_def_chains_star_inward() -> None:
    """Inward star graph hub has multiple definitions."""
    graph = star_graph(3, inward=True)  # spokes -> hub
    result = compute_use_def_chains(graph)

    # Hub has 3 definitions
    expect_equal(set(result["hub"]), {"spoke1", "spoke2", "spoke3"})


# ===========================================================================
# compute_dfg_density Tests
# ===========================================================================


def test_dfg_density_empty_graph_returns_zero() -> None:
    """Empty graph returns zero density."""
    graph = nx.DiGraph()
    result = compute_dfg_density(graph)
    expect_equal(result, DENSITY_ZERO)


def test_dfg_density_single_node_returns_zero() -> None:
    """Single node graph returns zero density."""
    graph = nx.DiGraph()
    graph.add_node("A")
    result = compute_dfg_density(graph)
    expect_equal(result, DENSITY_ZERO)


def test_dfg_density_complete_graph() -> None:
    """Complete directed graph has density 1.0."""
    graph = complete_digraph(4)
    result = compute_dfg_density(graph)

    # Complete digraph: n*(n-1) edges / n*(n-1) max = 1.0
    expected_density: float = 1.0
    expect_true(abs(result - expected_density) < DENSITY_TOLERANCE)


def test_dfg_density_chain_graph() -> None:
    """Chain graph has low density."""
    graph = chain_graph(4)  # 3 edges, max 12 edges
    result = compute_dfg_density(graph)

    # Density = 3 / 12 = 0.25
    expected_density: float = 0.25
    expect_true(abs(result - expected_density) < DENSITY_TOLERANCE)


def test_dfg_density_star_graph() -> None:
    """Star graph has moderate density."""
    graph = star_graph(3)  # 3 edges, 4 nodes, max 12 edges
    result = compute_dfg_density(graph)

    expected_density: float = 3 / 12
    expect_true(abs(result - expected_density) < DENSITY_TOLERANCE)


def test_dfg_density_two_node_graph() -> None:
    """Two node graph density calculation."""
    graph = nx.DiGraph([("A", "B")])
    result = compute_dfg_density(graph)

    # 1 edge, max 2 edges -> density 0.5
    expected_density: float = 0.5
    expect_true(abs(result - expected_density) < DENSITY_TOLERANCE)


# ===========================================================================
# find_dfg_cycles Tests
# ===========================================================================


def test_find_cycles_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
    result = find_dfg_cycles(graph)
    expect_equal(result, [])


def test_find_cycles_dag_returns_empty() -> None:
    """DAG has no cycles."""
    graph = chain_graph(5)
    result = find_dfg_cycles(graph)
    expect_equal(result, [])


def test_find_cycles_simple_cycle() -> None:
    """Simple cycle is detected."""
    graph = cyclic_graph(3)  # A -> B -> C -> A
    result = find_dfg_cycles(graph)

    expect_true(len(result) >= 1)
    # Cycle should contain all three nodes
    cycle_nodes = set(result[0])
    expect_equal(cycle_nodes, {"A", "B", "C"})


def test_find_cycles_self_loop() -> None:
    """Self-loop is detected as cycle."""
    graph = nx.DiGraph()
    graph.add_edge("A", "A")
    result = find_dfg_cycles(graph)

    expect_true(len(result) >= 1)
    expect_in("A", result[0])


def test_find_cycles_multiple_cycles() -> None:
    """Multiple cycles are detected."""
    graph = nx.DiGraph()
    # Cycle 1: A -> B -> A
    graph.add_edges_from([("A", "B"), ("B", "A")])
    # Cycle 2: C -> D -> C
    graph.add_edges_from([("C", "D"), ("D", "C")])

    result = find_dfg_cycles(graph)

    expect_true(len(result) >= EXPECTED_TWO_COMPONENTS)


def test_find_cycles_limit_respected() -> None:
    """Limit parameter caps number of cycles returned."""
    graph = nx.DiGraph()
    # Multiple small cycles
    for i in range(5):
        graph.add_edge(f"A{i}", f"B{i}")
        graph.add_edge(f"B{i}", f"A{i}")

    result = find_dfg_cycles(graph, limit=CYCLE_LIMIT_ONE)

    expect_true(len(result) <= CYCLE_LIMIT_ONE)


def test_find_cycles_nested_cycles() -> None:
    """Nested cycles are detected."""
    graph = nx.DiGraph()
    # Outer cycle: A -> B -> C -> A
    # Inner cycle: B -> D -> B
    graph.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("B", "D"),
            ("D", "B"),
        ]
    )

    result = find_dfg_cycles(graph)

    # Should find multiple cycles
    expect_true(len(result) >= EXPECTED_TWO_COMPONENTS)


def test_find_cycles_with_dag_part() -> None:
    """Graph with both cycle and DAG parts."""
    graph = nx.DiGraph()
    # DAG part
    graph.add_edges_from([("start", "A"), ("end", "exit")])
    # Cycle part
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # Connection
    graph.add_edge("C", "end")

    result = find_dfg_cycles(graph)

    # Should find the cycle
    expect_true(len(result) >= EXPECTED_SINGLE_COMPONENT)


# ===========================================================================
# Dataclass Frozen Tests
# ===========================================================================


def test_dfg_path_stats_frozen() -> None:
    """DFGPathStats is frozen."""
    stats = DFGPathStats(
        max_def_use_distance=3,
        avg_def_use_distance=2.0,
        reach_count=5,
    )
    assert_cannot_setattr(stats, "max_def_use_distance", 10)


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("chain_length", "expected_max_from_first"),
    [
        (2, 1),
        (3, 2),
        (5, 4),
        (10, 9),
    ],
)
def test_path_lengths_various_chains(chain_length: int, expected_max_from_first: int) -> None:
    """Chain graphs of various lengths have correct max distances."""
    graph = chain_graph(chain_length)
    result = compute_dfg_path_lengths(graph)

    expect_equal(result["A"].max_def_use_distance, expected_max_from_first)


@pytest.mark.parametrize(
    "cycle_size",
    [2, 3, 5, 10],
)
def test_cycles_detected_various_sizes(cycle_size: int) -> None:
    """Cycles of various sizes are detected."""
    graph = cyclic_graph(cycle_size)
    result = find_dfg_cycles(graph)

    expect_true(len(result) > 0)


@pytest.mark.parametrize(
    ("node_count", "edge_count", "expected_density"),
    [
        (3, 2, 2 / 6),  # 2 edges, max 6
        (4, 6, 6 / 12),  # 6 edges, max 12
        (5, 10, 10 / 20),  # 10 edges, max 20
    ],
)
def test_density_various_graphs(node_count: int, edge_count: int, expected_density: float) -> None:
    """Density calculation for various graph configurations."""
    graph = nx.DiGraph()
    for i in range(node_count):
        graph.add_node(i)
    edges_added = 0
    for i in range(node_count):
        for j in range(node_count):
            if i != j and edges_added < edge_count:
                graph.add_edge(i, j)
                edges_added += 1

    result = compute_dfg_density(graph)

    expect_true(abs(result - expected_density) < DENSITY_TOLERANCE)
