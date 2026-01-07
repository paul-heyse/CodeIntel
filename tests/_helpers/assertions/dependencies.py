"""Assertion helpers for dependency graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.graphs.compute.metrics.components import find_cycles
from codeintel.build.graphs.rx.algos import GraphInput, graph_edge_count
from codeintel.build.graphs.rx.store import RxGraphStore

from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.run_records import TargetRunRecord


def build_dependency_graph(edges: Iterable[tuple[str, str]]) -> RxGraphStore:
    """Create a directed graph from dependency edges.

    Parameters
    ----------
    edges
        Iterable of (source, target) tuples to add to the graph.

    Returns
    -------
    RxGraphStore
        Graph store containing the provided edges.
    """
    graph = RxGraphStore.directed()
    for src, dst in edges:
        graph.add_weighted_edge(src, dst, weight=1.0)
    return graph


def assert_edge_count(graph: GraphInput, expected: int) -> None:
    """Assert graph has expected number of edges.

    Parameters
    ----------
    graph
        Graph under validation.
    expected
        Expected edge count.
    """
    expect_equal(graph_edge_count(graph), expected)


def assert_cycle_count(graph: GraphInput, expected: int = 0) -> None:
    """Assert graph has the expected number of simple cycles.

    Parameters
    ----------
    graph
        Graph under validation.
    expected
        Expected number of cycles.
    """
    expect_equal(len(list(find_cycles(graph))), expected)


def assert_no_cycles(graph: GraphInput) -> None:
    """Assert graph has no cycles.

    Parameters
    ----------
    graph
        Graph under validation.
    """
    expect_true(not list(find_cycles(graph)))


def require_upstream_ok(
    record: TargetRunRecord,
    *,
    target: str,
    allow_skipped: bool = True,
) -> None:
    """Assert a dependency TargetRunRecord is acceptable for downstream use.

    Parameters
    ----------
    record
        Upstream target record to validate.
    target
        Target name for error context.
    allow_skipped
        Whether to treat skipped as cached success.

    Raises
    ------
    AssertionError
        If the upstream target status is not acceptable.
    """
    if record.status == "succeeded":
        return
    if allow_skipped and record.status == "skipped":
        return
    message = f"Upstream target {target} not ready (status={record.status})"
    raise AssertionError(message)


__all__ = [
    "assert_cycle_count",
    "assert_edge_count",
    "assert_no_cycles",
    "build_dependency_graph",
    "require_upstream_ok",
]
