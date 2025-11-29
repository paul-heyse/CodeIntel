"""Unit tests for repository-driven graph metric filters."""

from __future__ import annotations

import networkx as nx

from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters, build_graph_metric_filters
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.gateway import open_ingestion_gateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_filter_call_graph_prunes_nodes() -> None:
    """Call graph filter should restrict nodes to the provided GOIDs."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    filters = GraphMetricFilters(function_goids={1, 2})

    filtered = filters.filter_call_graph(graph)

    _expect(condition=set(filtered.nodes) == {1, 2}, detail="filtered nodes should match allowlist")
    _expect(
        condition=set(filtered.edges) == {(1, 2)},
        detail="filtered edges should match surviving nodes",
    )


def test_filter_import_graph_noop_without_modules() -> None:
    """Import graph filter should no-op when no modules are configured."""
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    filters = GraphMetricFilters(modules=None)

    filtered = filters.filter_import_graph(graph)

    _expect(
        condition=set(filtered.nodes) == {"a", "b"}, detail="import nodes should remain unchanged"
    )
    _expect(
        condition=set(filtered.edges) == {("a", "b")}, detail="import edges should remain unchanged"
    )


def test_build_filters_safe_when_repos_empty(tmp_path: object) -> None:
    """Building filters from empty repositories should yield no-op filters."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        cfg_snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)  # type: ignore[arg-type]
        cfg = GraphMetricsStepConfig(snapshot=cfg_snapshot)
        filters = build_graph_metric_filters(gateway, cfg)
        _expect(
            condition=filters.function_goids is None,
            detail="function filter should default to None",
        )
        _expect(condition=filters.modules is None, detail="module filter should default to None")
    finally:
        gateway.close()


def test_filter_subsystem_memberships_respects_allowlists() -> None:
    """Subsystem membership filtering should honor subsystem and module allowlists."""
    filters = GraphMetricFilters(
        subsystems={"s1"},
        modules={"mod.a"},
    )
    memberships = [("s1", "mod.a"), ("s1", "mod.b"), ("s2", "mod.a"), ("s3", "mod.c")]

    filtered = filters.filter_subsystem_memberships(memberships)

    _expect(
        condition=filtered == [("s1", "mod.a")],
        detail="only matching subsystem+module should remain",
    )


def test_filter_subsystem_graph_prunes_nodes() -> None:
    """Subsystem graph filter should restrict nodes to the provided allowlist."""
    graph = nx.DiGraph()
    graph.add_edge("s1", "s2")
    graph.add_edge("s2", "s3")
    filters = GraphMetricFilters(subsystems={"s1", "s2"})

    filtered = filters.filter_subsystem_graph(graph)

    _expect(
        condition=set(filtered.nodes) == {"s1", "s2"}, detail="subsystem nodes should be pruned"
    )
    _expect(
        condition=set(filtered.edges) == {("s1", "s2")}, detail="edges should match surviving nodes"
    )
