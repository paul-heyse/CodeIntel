"""Integration test for repository-backed graph metric filters on architecture seeds."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricFilters,
    GraphMetricsDeps,
    compute_graph_metrics,
)
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.architecture import open_seeded_architecture_gateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_filters_prune_architecture_metrics(tmp_path: Path) -> None:
    """Filters should restrict module/subsystem metrics derived from architecture data."""
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "arch.duckdb",
        strict_schema=True,
    )
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
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

    _expect(
        condition=modules == {"pkg.alpha"},
        detail=f"module metrics should honor filter allowlist; saw {modules}",
    )
    _expect(
        condition=modules_ext == {"pkg.alpha"},
        detail=f"module ext metrics should honor filter allowlist; saw {modules_ext}",
    )
    _expect(
        condition=subsystems == {"sub1"},
        detail=f"subsystem metrics should honor filter allowlist; saw {subsystems}",
    )
