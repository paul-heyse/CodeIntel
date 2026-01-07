"""Graph assertion helpers for analytics tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import rustworkx as rx

from codeintel.build.graphs.compute.metrics.components import (
    find_connected,
    find_cycles,
    find_strongly_connected,
    find_weakly_connected,
)
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ensure_directed_store,
    ensure_store,
    graph_edge_count,
    graph_node_count,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, stable_key
from codeintel.build.graphs.rx.store import RxGraphStore
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.config.primitives import SnapshotRef
    from tests._helpers.context import TestContext


@dataclass(frozen=True)
class GraphMetricsTableExpectations:
    """Expected row counts for graph metrics tables."""

    functions: int | None = None
    modules: int | None = None
    modules_min: int | None = None
    functions_ext: int | None = None
    modules_ext_min: int | None = None
    modules_ext: int | None = None
    config_keys: int | None = None
    config_projection_min: int | None = None
    graph_stats_min: int | None = None
    subsystem_metrics_min: int | None = None
    subsystem_agreement_min: int | None = None
    symbol_functions_min: int | None = None
    symbol_modules_min: int | None = None


@dataclass(frozen=True)
class FunctionMetricsExpectation:
    """Expected values for a graph_metrics_functions row."""

    goid: int
    fan_in: int
    fan_out: int
    in_degree: int
    out_degree: int
    cycle_member: bool


@dataclass(frozen=True)
class ModuleMetricsExpectation:
    """Expected values for a graph_metrics_modules row."""

    module: str
    import_fan_in: int
    import_fan_out: int
    symbol_fan_in: int
    symbol_fan_out: int
    import_cycle_member: bool


def assert_graph_counts(
    graph: GraphInput,
    *,
    nodes: int | None = None,
    edges: int | None = None,
) -> None:
    """Assert basic node/edge counts for a graph."""
    if nodes is not None:
        expect_equal(graph_node_count(graph), nodes)
    if edges is not None:
        expect_equal(graph_edge_count(graph), edges)


def assert_component_counts(
    graph: GraphInput,
    *,
    weak: int | None = None,
    strong: int | None = None,
) -> None:
    """Assert connected component counts for directed or undirected graphs."""
    store = ensure_store(graph)
    if weak is not None:
        if store.is_directed:
            expect_equal(len(tuple(find_weakly_connected(store))), weak)
        else:
            expect_equal(len(tuple(find_connected(store))), weak)
    if strong is not None:
        if store.is_directed:
            expect_equal(len(find_strongly_connected(store).components), strong)
        else:
            expect_equal(len(tuple(find_connected(store))), strong)


def assert_cycle_membership(graph: GraphInput, expected: Iterable[Iterable[object]]) -> None:
    """Assert that a directed graph contains the expected simple cycles."""
    cycles = [tuple(cycle) for cycle in find_cycles(graph)]
    expect_equal(set(map(tuple, expected)), set(cycles))


def assert_filtered_graph(
    graph: GraphInput,
    *,
    expected_nodes: Collection[object],
    expected_edges: Collection[tuple[object, object]],
) -> None:
    """Assert nodes and edges on a filtered directed graph."""
    store = ensure_store(graph)
    node_set = set(expected_nodes)
    edge_set = set(expected_edges)
    actual_nodes = set(store.node_ids())
    actual_edges = {
        (store.index_to_id[src_idx], store.index_to_id[dst_idx])
        for src_idx, dst_idx in store.graph.edge_list()
    }
    expect_equal(actual_nodes, node_set)
    expect_equal(actual_edges, edge_set)
    expect_true(edge_set <= actual_edges)


def assert_graph_metrics_for_goids(ctx: TestContext, goids: Iterable[int]) -> None:
    """Assert function graph metrics exist for the provided GOIDs."""
    for goid in goids:
        count = ctx.query_count(
            "analytics.graph_metrics_functions",
            f"function_goid_h128 = {goid}",
        )
        expect_true(count > 0, message=f"Expected graph metrics for GOID {goid}")


def assert_cycle_counts(graph: GraphInput, expected: int) -> None:
    """Assert the directed graph contains the expected number of simple cycles."""
    expect_equal(len(tuple(find_cycles(graph))), expected)


def _count_for_snapshot(con: DuckDBPyConnection, snapshot: SnapshotRef, query: str) -> int:
    result = con.execute(query, [snapshot.repo, snapshot.commit]).fetchone()
    expect_is_not_none(result)
    first = result[0] if result else 0
    return int(first)


def _expect_count_equal_if_present(
    con: DuckDBPyConnection, snapshot: SnapshotRef, *, query: str, expected: int | None
) -> None:
    if expected is None:
        return
    expect_equal(_count_for_snapshot(con, snapshot, query), expected)


def _expect_count_at_least_if_present(
    con: DuckDBPyConnection, snapshot: SnapshotRef, *, query: str, minimum: int | None
) -> None:
    if minimum is None:
        return
    expect_true(_count_for_snapshot(con, snapshot, query) >= minimum)


def assert_graph_metrics_table_counts(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    expectations: GraphMetricsTableExpectations,
) -> None:
    """Assert core graph metrics table counts for a snapshot."""
    _expect_count_equal_if_present(
        con,
        snapshot,
        query="SELECT COUNT(*) FROM analytics.config_graph_metrics_keys WHERE repo = ? AND commit = ?",
        expected=expectations.config_keys,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.config_projection_module_edges "
            "WHERE repo = ? AND commit = ?"
        ),
        minimum=expectations.config_projection_min,
    )
    _expect_count_equal_if_present(
        con,
        snapshot,
        query="SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        expected=expectations.functions,
    )
    _expect_count_equal_if_present(
        con,
        snapshot,
        query="SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        expected=expectations.modules,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query="SELECT COUNT(*) FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        minimum=expectations.modules_min,
    )
    _expect_count_equal_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.graph_metrics_functions_ext "
            "WHERE repo = ? AND commit = ?"
        ),
        expected=expectations.functions_ext,
    )
    _expect_count_equal_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?"
        ),
        expected=expectations.modules_ext,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?"
        ),
        minimum=expectations.modules_ext_min,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query="SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        minimum=expectations.graph_stats_min,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?"
        ),
        minimum=expectations.subsystem_metrics_min,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=("SELECT COUNT(*) FROM analytics.subsystem_agreement WHERE repo = ? AND commit = ?"),
        minimum=expectations.subsystem_agreement_min,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules "
            "WHERE repo = ? AND commit = ?"
        ),
        minimum=expectations.symbol_modules_min,
    )
    _expect_count_at_least_if_present(
        con,
        snapshot,
        query=(
            "SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions "
            "WHERE repo = ? AND commit = ?"
        ),
        minimum=expectations.symbol_functions_min,
    )


def assert_graph_metrics_function_row(
    con: DuckDBPyConnection,
    expectation: FunctionMetricsExpectation,
) -> None:
    """Assert a graph_metrics_functions row matches expected values."""
    row = con.execute(
        """
        SELECT call_fan_in, call_fan_out, call_in_degree, call_out_degree, call_cycle_member
        FROM analytics.graph_metrics_functions
        WHERE function_goid_h128 = ?
        """,
        [expectation.goid],
    ).fetchone()
    expect_is_not_none(
        row, message=f"Missing graph_metrics_functions row for GOID {expectation.goid}"
    )
    if row is None:
        return
    expect_equal(
        tuple(row),
        (
            expectation.fan_in,
            expectation.fan_out,
            expectation.in_degree,
            expectation.out_degree,
            expectation.cycle_member,
        ),
    )


def assert_graph_metrics_module_row(
    con: DuckDBPyConnection,
    expectation: ModuleMetricsExpectation,
) -> None:
    """Assert a graph_metrics_modules row matches expected values."""
    row = con.execute(
        """
        SELECT import_fan_in, import_fan_out, symbol_fan_in, symbol_fan_out, import_cycle_member
        FROM analytics.graph_metrics_modules
        WHERE module = ?
        """,
        [expectation.module],
    ).fetchone()
    expect_is_not_none(row, message=f"Missing graph_metrics_modules row for {expectation.module}")
    if row is None:
        return
    expect_equal(
        tuple(row),
        (
            expectation.import_fan_in,
            expectation.import_fan_out,
            expectation.symbol_fan_in,
            expectation.symbol_fan_out,
            expectation.import_cycle_member,
        ),
    )


def expect_graph_equal(
    actual: GraphInput,
    expected: GraphInput,
    *,
    message: str | None = None,
) -> None:
    """Assert that two graphs have identical nodes and edges (including attributes)."""
    actual_store = ensure_store(actual)
    expected_store = ensure_store(expected)
    node_label = message or "graph_nodes"
    edge_label = message or "graph_edges"
    expect_equal(
        _node_payload(actual_store, include_attrs=True),
        _node_payload(expected_store, include_attrs=True),
        label=node_label,
    )
    expect_equal(
        _edge_payload(actual_store, include_attrs=True),
        _edge_payload(expected_store, include_attrs=True),
        label=edge_label,
    )


def expect_same_nodes_edges(
    actual: GraphInput,
    expected: GraphInput,
    *,
    node_attrs: bool = True,
    edge_attrs: bool = True,
    message: str | None = None,
) -> None:
    """Assert graphs share the same nodes/edges, optionally ignoring attributes."""
    actual_store = ensure_store(actual)
    expected_store = ensure_store(expected)
    label = message or "graph"
    expect_equal(
        _node_payload(actual_store, include_attrs=node_attrs),
        _node_payload(expected_store, include_attrs=node_attrs),
        label=f"{label}_nodes",
    )
    expect_equal(
        _edge_payload(actual_store, include_attrs=edge_attrs),
        _edge_payload(expected_store, include_attrs=edge_attrs),
        label=f"{label}_edges",
    )


def expect_graph_is_dag(graph: GraphInput, *, message: str | None = None) -> None:
    """Assert that a directed graph is a DAG."""
    store = ensure_directed_store(graph)
    directed = cast("rx.PyDiGraph", store.graph)
    expect_true(
        rx.is_directed_acyclic_graph(directed),
        message=message or "Expected DAG",
    )


def expect_has_cycle(graph: GraphInput, *, message: str | None = None) -> None:
    """Assert that a directed graph contains at least one cycle."""
    store = ensure_directed_store(graph)
    directed = cast("rx.PyDiGraph", store.graph)
    expect_true(
        not rx.is_directed_acyclic_graph(directed),
        message=message or "Expected graph to contain a cycle",
    )


def require_projection_graph(
    graph: GraphInput | None,
    *,
    message: str | None = None,
) -> RxGraphStore:
    """Ensure a projection graph exists and return it.

    Returns
    -------
    RxGraphStore
        The provided graph store when present.

    Raises
    ------
    AssertionError
        If the projection graph is ``None``.
    """
    if graph is None:
        raise AssertionError(message or "Expected projection graph")
    return ensure_store(graph)


def _node_payload(
    store: RxGraphStore,
    *,
    include_attrs: bool,
) -> set[object] | set[tuple[object, tuple[tuple[str, object], ...]]]:
    if not include_attrs:
        return set(store.node_ids())
    return {
        (node_id, tuple(sorted(store.node_attrs.get(node_id, {}).items())))
        for node_id in store.node_ids()
    }


def _edge_payload(
    store: RxGraphStore,
    *,
    include_attrs: bool,
) -> set[tuple[object, object]] | set[tuple[object, object, float]]:
    if include_attrs:
        detailed: set[tuple[object, object, float]] = set()
        for src_idx, dst_idx in store.graph.edge_list():
            src_id = store.index_to_id[src_idx]
            dst_id = store.index_to_id[dst_idx]
            left, right = _edge_key(store, src_id, dst_id)
            payload = store.graph.get_edge_data(src_idx, dst_idx)
            detailed.add((left, right, edge_weight_from_payload(payload)))
        return detailed
    simple: set[tuple[object, object]] = set()
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        left, right = _edge_key(store, src_id, dst_id)
        simple.add((left, right))
    return simple


def _edge_key(store: RxGraphStore, left: object, right: object) -> tuple[object, object]:
    if store.is_directed:
        return (left, right)
    ordered = sorted((left, right), key=stable_key)
    return (ordered[0], ordered[1])


__all__ = [
    "FunctionMetricsExpectation",
    "GraphMetricsTableExpectations",
    "ModuleMetricsExpectation",
    "assert_component_counts",
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
