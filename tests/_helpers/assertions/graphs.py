"""Graph assertion helpers for analytics tests."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def assert_graph_counts(
    graph: nx.Graph | nx.DiGraph,
    *,
    nodes: int | None = None,
    edges: int | None = None,
) -> None:
    """Assert basic node/edge counts for a graph."""
    if nodes is not None:
        expect_equal(graph.number_of_nodes(), nodes)
    if edges is not None:
        expect_equal(graph.number_of_edges(), edges)


def assert_component_counts(
    graph: nx.Graph | nx.DiGraph,
    *,
    weak: int | None = None,
    strong: int | None = None,
) -> None:
    """Assert connected component counts for directed or undirected graphs."""
    if weak is not None:
        if isinstance(graph, nx.DiGraph):
            expect_equal(len(tuple(nx.weakly_connected_components(graph))), weak)
        else:
            expect_equal(len(tuple(nx.connected_components(graph))), weak)
    if strong is not None:
        if isinstance(graph, nx.DiGraph):
            expect_equal(len(tuple(nx.strongly_connected_components(graph))), strong)
        else:
            expect_equal(len(tuple(nx.connected_components(graph))), strong)


def assert_cycle_membership(graph: nx.DiGraph, expected: Iterable[Iterable[object]]) -> None:
    """Assert that a directed graph contains the expected simple cycles."""
    cycles = [tuple(cycle) for cycle in nx.simple_cycles(graph)]
    expect_equal(set(map(tuple, expected)), set(cycles))


def assert_filtered_graph(
    graph: nx.DiGraph,
    *,
    expected_nodes: set[object],
    expected_edges: set[tuple[object, object]],
) -> None:
    """Assert nodes and edges on a filtered directed graph."""
    expect_equal(set(graph.nodes), expected_nodes)
    expect_equal(set(graph.edges), expected_edges)
    expect_true(expected_edges <= set(graph.edges))


__all__ = [
    "assert_component_counts",
    "assert_cycle_membership",
    "assert_filtered_graph",
    "assert_graph_counts",
]
