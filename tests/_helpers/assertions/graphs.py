"""Graph assertion helpers for analytics tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import networkx as nx

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


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


def assert_graph_metrics_for_goids(ctx: TestContext, goids: Iterable[int]) -> None:
    """Assert function graph metrics exist for the provided GOIDs."""
    for goid in goids:
        count = ctx.query_count(
            "analytics.graph_metrics_functions",
            f"function_goid_h128 = {goid}",
        )
        expect_true(count > 0, message=f"Expected graph metrics for GOID {goid}")


def assert_cycle_counts(graph: nx.DiGraph, expected: int) -> None:
    """Assert the directed graph contains the expected number of simple cycles."""
    expect_equal(len(tuple(nx.simple_cycles(graph))), expected)


def assert_coverage_ratio_between(
    ctx: TestContext, goid: int, *, low: float, high: float
) -> None:
    """Assert coverage ratio for a GOID falls within bounds."""
    row = ctx.query(
        """
        SELECT coverage_ratio FROM analytics.coverage_functions
        WHERE function_goid_h128 = ?
        """,
        [goid],
    )[0]
    ratio = float(cast("float", row.coverage_ratio))
    expect_true(low <= ratio <= high, message=f"coverage_ratio {ratio} outside [{low}, {high}]")


def expect_graph_equal(
    actual: nx.Graph | nx.DiGraph,
    expected: nx.Graph | nx.DiGraph,
    *,
    message: str | None = None,
) -> None:
    """Assert that two graphs have identical nodes and edges (including attributes)."""

    def node_payload(graph: nx.Graph | nx.DiGraph) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
        return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}

    def edge_payload(graph: nx.Graph | nx.DiGraph) -> set[tuple[object, object, tuple[tuple[str, object], ...]]]:
        return {
            (src, dst, tuple(sorted(data.items())))
            for src, dst, data in graph.edges(data=True)
        }

    node_label = message or "graph_nodes"
    edge_label = message or "graph_edges"
    expect_equal(node_payload(actual), node_payload(expected), label=node_label)
    expect_equal(edge_payload(actual), edge_payload(expected), label=edge_label)


def expect_same_nodes_edges(
    actual: nx.Graph | nx.DiGraph,
    expected: nx.Graph | nx.DiGraph,
    *,
    node_attrs: bool = True,
    edge_attrs: bool = True,
    message: str | None = None,
) -> None:
    """Assert graphs share the same nodes/edges, optionally ignoring attributes."""

    def nodes(graph: nx.Graph | nx.DiGraph) -> set[object] | set[tuple[object, tuple[tuple[str, object], ...]]]:
        if not node_attrs:
            return set(graph.nodes)
        return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}

    def edges(graph: nx.Graph | nx.DiGraph) -> set[tuple[object, object]] | set[
        tuple[object, object, tuple[tuple[str, object], ...]]
    ]:
        if not edge_attrs:
            return set(graph.edges)
        return {
            (src, dst, tuple(sorted(data.items())))
            for src, dst, data in graph.edges(data=True)
        }

    label = message or "graph"
    expect_equal(nodes(actual), nodes(expected), label=f"{label}_nodes")
    expect_equal(edges(actual), edges(expected), label=f"{label}_edges")


def expect_graph_is_dag(graph: nx.DiGraph, *, message: str | None = None) -> None:
    """Assert that a directed graph is a DAG."""
    expect_true(nx.is_directed_acyclic_graph(graph), message=message or "Expected DAG")


def expect_has_cycle(graph: nx.DiGraph, *, message: str | None = None) -> None:
    """Assert that a directed graph contains at least one cycle."""
    expect_true(
        not nx.is_directed_acyclic_graph(graph),
        message=message or "Expected graph to contain a cycle",
    )


def require_projection_graph(graph: nx.Graph | None, *, message: str | None = None) -> nx.Graph:
    """Ensure a projection graph exists and return it.

    Returns
    -------
    nx.Graph
        The provided graph when present.

    Raises
    ------
    AssertionError
        If the projection graph is ``None``.
    """
    expect_is_not_none(graph, message=message or "Expected projection graph")
    if graph is None:
        raise AssertionError(message or "Expected projection graph")
    return graph


__all__ = [
    "assert_component_counts",
    "assert_coverage_ratio_between",
    "assert_cycle_counts",
    "assert_cycle_membership",
    "assert_filtered_graph",
    "assert_graph_counts",
    "assert_graph_metrics_for_goids",
    "expect_graph_equal",
    "expect_graph_is_dag",
    "expect_has_cycle",
    "expect_same_nodes_edges",
    "require_projection_graph",
]
