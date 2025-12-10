"""Assertion helpers for dependency graphs."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def build_dependency_graph(edges: Iterable[tuple[str, str]]) -> nx.DiGraph:
    """Create a directed graph from dependency edges.

    Parameters
    ----------
    edges
        Iterable of (source, target) tuples to add to the graph.

    Returns
    -------
    nx.DiGraph
        Graph containing the provided edges.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def assert_edge_count(graph: nx.DiGraph, expected: int) -> None:
    """Assert graph has expected number of edges.

    Parameters
    ----------
    graph
        Graph under validation.
    expected
        Expected edge count.
    """
    expect_equal(graph.number_of_edges(), expected)


def assert_cycle_count(graph: nx.DiGraph, expected: int = 0) -> None:
    """Assert graph has the expected number of simple cycles.

    Parameters
    ----------
    graph
        Graph under validation.
    expected
        Expected number of cycles.
    """
    expect_equal(len(list(nx.simple_cycles(graph))), expected)


def assert_no_cycles(graph: nx.DiGraph) -> None:
    """Assert graph has no cycles.

    Parameters
    ----------
    graph
        Graph under validation.
    """
    expect_true(not list(nx.simple_cycles(graph)))


__all__ = [
    "assert_cycle_count",
    "assert_edge_count",
    "assert_no_cycles",
    "build_dependency_graph",
]
