"""Integration and unit coverage for graph metric filters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters, build_graph_metric_filters
from tests._helpers import TestScenario
from tests._helpers.assertions import (
    assert_component_counts,
    assert_filtered_graph,
    assert_graph_counts,
    expect_equal,
)
from tests._helpers.factories import make_snapshot
from tests._helpers.fixtures.graphs import chain_graph, cyclic_graph, disconnected_graph
from tests._helpers.graph_runtime_harness import (
    run_graph_metrics_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.graph_runtime_harness import (
        GraphRuntimeHarness,
    )


def test_filters_prune_metrics(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Filters should restrict module and subsystem metrics from the canonical sample."""
    filters = GraphMetricFilters(
        function_goids={
            graph_runtime_ctx.goids["func_a"],
            graph_runtime_ctx.goids["func_b"],
        },
        modules={"pkg.mod_a", "pkg.service"},
        subsystems={"core"},
    )

    run_graph_metrics_pipeline(graph_runtime_ctx, filters=filters)

    params = [graph_runtime_ctx.snapshot.repo, graph_runtime_ctx.snapshot.commit]
    gateway = graph_runtime_ctx.gateway
    modules = {
        row[0]
        for row in gateway.con.execute(
            "SELECT module FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
            params,
        ).fetchall()
    }
    modules_ext = {
        row[0]
        for row in gateway.con.execute(
            "SELECT module FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
            params,
        ).fetchall()
    }
    subsystems = {
        row[0]
        for row in gateway.con.execute(
            "SELECT subsystem_id FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
            params,
        ).fetchall()
    }
    functions = {
        row[0]
        for row in gateway.con.execute(
            "SELECT function_goid_h128 FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
            params,
        ).fetchall()
    }

    expect_equal(modules, {"pkg.mod_a"})
    expect_equal(modules_ext, {"pkg.mod_a"})
    expect_equal(subsystems, {"core"})
    expect_equal(
        functions,
        {
            graph_runtime_ctx.goids["func_a"],
            graph_runtime_ctx.goids["func_b"],
        },
    )


def test_filter_call_graph_prunes_nodes() -> None:
    """Call graph filter should restrict nodes to the provided GOIDs."""
    graph = chain_graph(3)
    allowed_nodes = set(list(graph.nodes)[:2])
    filters = GraphMetricFilters(function_goids=allowed_nodes)

    filtered = filters.filter_call_graph(graph)

    assert_filtered_graph(filtered, expected_nodes=allowed_nodes, expected_edges={("A", "B")})
    assert_component_counts(filtered, weak=1, strong=2)
    assert_graph_counts(filtered, nodes=2, edges=1)


def test_filter_import_graph_noop_without_modules() -> None:
    """Import graph filter should no-op when no modules are configured."""
    graph = chain_graph(2)
    filters = GraphMetricFilters(modules=None)

    filtered = filters.filter_import_graph(graph)

    assert_filtered_graph(
        filtered, expected_nodes=set(graph.nodes), expected_edges=set(graph.edges)
    )
    assert_component_counts(filtered, weak=1, strong=2)


def test_build_filters_safe_when_repos_empty(tmp_path: Path) -> None:
    """Building filters from empty repositories should yield no-op filters."""
    ctx = TestScenario.empty().build(tmp_path)
    snapshot = make_snapshot(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
    filters = build_graph_metric_filters(ctx.gateway, snapshot)
    expect_equal(filters.function_goids, None)
    expect_equal(filters.modules, None)
    ctx.close()


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
    graph = cyclic_graph(3)
    filters = GraphMetricFilters(subsystems={"A", "B"})

    filtered = filters.filter_subsystem_graph(graph)

    assert_filtered_graph(filtered, expected_nodes={"A", "B"}, expected_edges={("A", "B")})
    assert_component_counts(filtered, weak=1, strong=2)


def test_filter_call_graph_preserves_component_counts() -> None:
    """Filtering a disconnected graph should preserve component totals."""
    graph = disconnected_graph()
    filters = GraphMetricFilters(function_goids=set(graph.nodes))

    filtered = filters.filter_call_graph(graph)

    assert_graph_counts(filtered, nodes=6, edges=4)
    assert_component_counts(filtered, weak=2, strong=6)
