"""Integration test for GraphServiceRuntime orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions, GraphServiceRuntime
from codeintel.analytics.graphs.plugins import DEFAULT_GRAPH_METRIC_PLUGINS
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.architecture import open_seeded_architecture_gateway


@pytest.mark.integration
def test_graph_service_runtime_runs_default_plugins() -> None:
    """GraphServiceRuntime executes default plugins end-to-end."""
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
    service.run_plugins(DEFAULT_GRAPH_METRIC_PLUGINS, cfg=cfg)

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


@pytest.mark.integration
def test_graph_service_runtime_disables_plugins() -> None:
    """Disabling plugins via config should prevent their execution."""
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
    con = gateway.con
    con.execute(
        "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    cfg = GraphMetricsStepConfig(
        snapshot=snapshot,
        disabled_plugins=("core_graph_metrics",),
    )
    disabled = {"core_graph_metrics"}
    plugin_names = tuple(name for name in DEFAULT_GRAPH_METRIC_PLUGINS if name not in disabled)
    service.run_plugins(plugin_names, cfg=cfg)

    row = con.execute(
        "SELECT COUNT(*) FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    if row is None:
        pytest.fail("graph_metrics_functions returned no rows")
    if row[0] != 0:
        pytest.fail(f"Expected graph_metrics_functions to remain empty, saw {row[0]}")

    stats_row = con.execute(
        "SELECT COUNT(*) FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    if stats_row is None:
        pytest.fail("graph_stats returned no rows")
    if stats_row[0] <= 0:
        pytest.fail("graph_stats should be populated when other plugins run")


@pytest.mark.integration
def test_graph_service_runtime_writes_manifest(tmp_path: Path) -> None:
    """GraphServiceRuntime writes a manifest when requested."""
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
    manifest_path = tmp_path / "graph_plugin_manifest.json"

    report = service.run_plugins(
        DEFAULT_GRAPH_METRIC_PLUGINS,
        cfg=cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )

    if not manifest_path.exists():
        pytest.fail("Expected manifest file to be written")
    payload = json.loads(manifest_path.read_text())
    if payload.get("repo") != repo or payload.get("commit") != commit:
        pytest.fail("Manifest did not include repo/commit")
    if len(report.records) == 0:
        pytest.fail("Plugin run report should contain records")
    for record in report.records:
        if record.input_hash is None:
            pytest.fail("Expected input_hash to be recorded in manifest")
