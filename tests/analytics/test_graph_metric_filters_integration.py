"""Integration test for repository-backed graph metric filters on architecture seeds."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricFilters,
    GraphMetricsDeps,
    compute_graph_metrics,
)
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.factories import make_snapshot
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
