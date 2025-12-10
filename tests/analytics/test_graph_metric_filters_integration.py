"""Integration test for repository-backed graph metric filters on architecture seeds."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricFilters,
    GraphMetricsDeps,
    build_graph_metric_filters,
    compute_graph_metrics,
)
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import GatewayFactory
from tests._helpers.seeds.architecture import open_seeded_architecture_gateway


def test_filters_prune_architecture_metrics(tmp_path: Path) -> None:
    """Filters should restrict module/subsystem metrics derived from architecture data."""
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "arch.duckdb",
        strict_schema=True,
    )
    snapshot = make_snapshot(repo_root=tmp_path)
    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    runtime = build_graph_runtime(gateway, GraphRuntimeOptions(snapshot=snapshot))
    filters = GraphMetricFilters(
        modules={"pkg.alpha"},
        subsystems={"sub1"},
    )

    compute_graph_metrics(
        gateway,
        cfg,
        deps=GraphMetricsDeps(runtime=runtime, filters=filters),
    )
    compute_graph_metrics_functions_ext(
        gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
        runtime=runtime,
        filters=filters,
    )
    compute_graph_metrics_modules_ext(
        gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
        runtime=runtime,
        filters=filters,
    )
    compute_subsystem_graph_metrics(
        gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
        runtime=runtime,
        filters=filters,
    )

    modules = {
        row[0]
        for row in gateway.con.execute(
            "SELECT module FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
            [snapshot.repo, snapshot.commit],
        ).fetchall()
    }
    modules_ext = {
        row[0]
        for row in gateway.con.execute(
            "SELECT module FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
            [snapshot.repo, snapshot.commit],
        ).fetchall()
    }
    subsystems = {
        row[0]
        for row in gateway.con.execute(
            "SELECT subsystem_id FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
            [snapshot.repo, snapshot.commit],
        ).fetchall()
    }

    expect_equal(modules, {"pkg.alpha"})
    expect_equal(modules_ext, {"pkg.alpha"})
    expect_equal(subsystems, {"sub1"})


def test_filter_call_graph_prunes_nodes() -> None:
    """Call graph filter should restrict nodes to the provided GOIDs."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    filters = GraphMetricFilters(function_goids={1, 2})

    filtered = filters.filter_call_graph(graph)

    expect_equal(set(filtered.nodes), {1, 2})
    expect_equal(set(filtered.edges), {(1, 2)})


def test_filter_import_graph_noop_without_modules() -> None:
    """Import graph filter should no-op when no modules are configured."""
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    filters = GraphMetricFilters(modules=None)

    filtered = filters.filter_import_graph(graph)

    expect_equal(set(filtered.nodes), {"a", "b"})
    expect_equal(set(filtered.edges), {("a", "b")})


def test_build_filters_safe_when_repos_empty(tmp_path: Path) -> None:
    """Building filters from empty repositories should yield no-op filters."""
    gateway = GatewayFactory().with_views().open()
    try:
        cfg_snapshot = make_snapshot(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
        cfg = GraphMetricsStepConfig(snapshot=cfg_snapshot)
        filters = build_graph_metric_filters(gateway, cfg)
        expect_equal(filters.function_goids, None)
        expect_equal(filters.modules, None)
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

    expect_equal(filtered, [("s1", "mod.a")])


def test_filter_subsystem_graph_prunes_nodes() -> None:
    """Subsystem graph filter should restrict nodes to the provided allowlist."""
    graph = nx.DiGraph()
    graph.add_edge("s1", "s2")
    graph.add_edge("s2", "s3")
    filters = GraphMetricFilters(subsystems={"s1", "s2"})

    filtered = filters.filter_subsystem_graph(graph)

    expect_equal(set(filtered.nodes), {"s1", "s2"})
    expect_equal(set(filtered.edges), {("s1", "s2")})
