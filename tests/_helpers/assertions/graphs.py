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
    from duckdb import DuckDBPyConnection

    from codeintel.config.primitives import SnapshotRef
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


def assert_coverage_ratio_between(ctx: TestContext, goid: int, *, low: float, high: float) -> None:
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


def assert_graph_metrics_table_counts(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    *,
    functions: int | None = None,
    modules: int | None = None,
    modules_min: int | None = None,
    functions_ext: int | None = None,
    modules_ext_min: int | None = None,
    modules_ext: int | None = None,
    config_keys: int | None = None,
    config_projection_min: int | None = None,
    graph_stats_min: int | None = None,
    subsystem_metrics_min: int | None = None,
    subsystem_agreement_min: int | None = None,
    symbol_functions_min: int | None = None,
    symbol_modules_min: int | None = None,
) -> None:
    """Assert core graph metrics table counts for a snapshot."""

    def _count(query: str) -> int:
        result = con.execute(query, [snapshot.repo, snapshot.commit]).fetchone()
        return int(result[0])  # type: ignore[index]

    if config_keys is not None:
        expect_equal(
            _count(
                "SELECT COUNT(*) FROM analytics.config_graph_metrics_keys "
                "WHERE repo = ? AND commit = ?"
            ),
            config_keys,
        )
    if config_projection_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.config_projection_module_edges "
                "WHERE repo = ? AND commit = ?"
            )
            >= config_projection_min
        )
    if functions is not None:
        expect_equal(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_functions "
                "WHERE repo = ? AND commit = ?"
            ),
            functions,
        )
    if modules is not None:
        expect_equal(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?"
            ),
            modules,
        )
    if modules_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?"
            )
            >= modules_min
        )
    if functions_ext is not None:
        expect_equal(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_functions_ext "
                "WHERE repo = ? AND commit = ?"
            ),
            functions_ext,
        )
    if modules_ext is not None:
        expect_equal(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext "
                "WHERE repo = ? AND commit = ?"
            ),
            modules_ext,
        )
    if modules_ext_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext "
                "WHERE repo = ? AND commit = ?"
            )
            >= modules_ext_min
        )
    if graph_stats_min is not None:
        expect_true(
            _count("SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?")
            >= graph_stats_min
        )
    if subsystem_metrics_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.subsystem_graph_metrics "
                "WHERE repo = ? AND commit = ?"
            )
            >= subsystem_metrics_min
        )
    if subsystem_agreement_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.subsystem_agreement WHERE repo = ? AND commit = ?"
            )
            >= subsystem_agreement_min
        )
    if symbol_modules_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules "
                "WHERE repo = ? AND commit = ?"
            )
            >= symbol_modules_min
        )
    if symbol_functions_min is not None:
        expect_true(
            _count(
                "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions "
                "WHERE repo = ? AND commit = ?"
            )
            >= symbol_functions_min
        )


def assert_graph_metrics_function_row(
    con: DuckDBPyConnection,
    *,
    goid: int,
    fan_in: int,
    fan_out: int,
    in_degree: int,
    out_degree: int,
    cycle_member: bool,
) -> None:
    """Assert a graph_metrics_functions row matches expected values."""
    row = con.execute(
        """
        SELECT call_fan_in, call_fan_out, call_in_degree, call_out_degree, call_cycle_member
        FROM analytics.graph_metrics_functions
        WHERE function_goid_h128 = ?
        """,
        [goid],
    ).fetchone()
    expect_is_not_none(row, message=f"Missing graph_metrics_functions row for GOID {goid}")
    if row is None:
        return
    expect_equal(tuple(row), (fan_in, fan_out, in_degree, out_degree, cycle_member))


def assert_graph_metrics_module_row(
    con: DuckDBPyConnection,
    *,
    module: str,
    import_fan_in: int,
    import_fan_out: int,
    symbol_fan_in: int,
    symbol_fan_out: int,
    import_cycle_member: bool,
) -> None:
    """Assert a graph_metrics_modules row matches expected values."""
    row = con.execute(
        """
        SELECT import_fan_in, import_fan_out, symbol_fan_in, symbol_fan_out, import_cycle_member
        FROM analytics.graph_metrics_modules
        WHERE module = ?
        """,
        [module],
    ).fetchone()
    expect_is_not_none(row, message=f"Missing graph_metrics_modules row for {module}")
    if row is None:
        return
    expect_equal(
        tuple(row),
        (import_fan_in, import_fan_out, symbol_fan_in, symbol_fan_out, import_cycle_member),
    )


def expect_graph_equal(
    actual: nx.Graph | nx.DiGraph,
    expected: nx.Graph | nx.DiGraph,
    *,
    message: str | None = None,
) -> None:
    """Assert that two graphs have identical nodes and edges (including attributes)."""

    def node_payload(
        graph: nx.Graph | nx.DiGraph,
    ) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
        return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}

    def edge_payload(
        graph: nx.Graph | nx.DiGraph,
    ) -> set[tuple[object, object, tuple[tuple[str, object], ...]]]:
        return {
            (src, dst, tuple(sorted(data.items()))) for src, dst, data in graph.edges(data=True)
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

    def nodes(
        graph: nx.Graph | nx.DiGraph,
    ) -> set[object] | set[tuple[object, tuple[tuple[str, object], ...]]]:
        if not node_attrs:
            return set(graph.nodes)
        return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}

    def edges(
        graph: nx.Graph | nx.DiGraph,
    ) -> set[tuple[object, object]] | set[tuple[object, object, tuple[tuple[str, object], ...]]]:
        if not edge_attrs:
            return set(graph.edges)
        return {
            (src, dst, tuple(sorted(data.items()))) for src, dst, data in graph.edges(data=True)
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
