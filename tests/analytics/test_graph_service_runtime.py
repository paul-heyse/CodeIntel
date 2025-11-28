"""Integration test for GraphServiceRuntime orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import GraphServiceRuntime
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.architecture import open_seeded_architecture_gateway


@pytest.mark.integration
def test_graph_service_runtime_end_to_end() -> None:
    """GraphServiceRuntime computes core graph metrics end-to-end."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = open_seeded_architecture_gateway(repo=repo, commit=commit)
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
    )
    service = GraphServiceRuntime(gateway=gateway, runtime=runtime)

    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    service.compute_graph_metrics(cfg)
    service.compute_graph_metrics_ext(repo=repo, commit=commit)
    service.compute_symbol_metrics(repo=repo, commit=commit)
    service.compute_subsystem_metrics(repo=repo, commit=commit)
    service.compute_graph_stats(repo=repo, commit=commit)

    con = gateway.con
    checks = [
        (
            "analytics.graph_metrics_functions",
            "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        ),
        (
            "analytics.graph_metrics_modules_ext",
            "SELECT COUNT(*) FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
        ),
        (
            "analytics.subsystem_graph_metrics",
            "SELECT COUNT(*) FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
        ),
        (
            "analytics.graph_stats",
            "SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        ),
    ]
    for table, sql in checks:
        row = con.execute(sql, [repo, commit]).fetchone()
        if row is None:
            pytest.fail(f"{table} returned no rows")
        count = row[0]
        if count is None or count <= 0:
            pytest.fail(f"{table} is empty after graph service runtime execution")
