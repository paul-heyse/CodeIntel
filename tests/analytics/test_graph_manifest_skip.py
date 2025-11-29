"""Tests for manifest-based skip and dry-run behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions, GraphServiceRuntime
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy, GraphRunScope
from codeintel.storage.gateway import open_memory_gateway


@pytest.fixture(name="snapshot_repo_commit")
def _snapshot_repo_commit() -> tuple[str, str, SnapshotRef]:
    repo = "demo/repo"
    commit = "deadbeef"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    return repo, commit, snapshot


def _build_service(snapshot: SnapshotRef) -> GraphServiceRuntime:
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
    )
    return GraphServiceRuntime(gateway=gateway, runtime=runtime)


def test_skip_on_unchanged_reads_manifest(
    tmp_path: Path, snapshot_repo_commit: tuple[str, str, SnapshotRef]
) -> None:
    """Skip-on-unchanged should skip a plugin when manifest matches input hash."""
    _, _, snapshot = snapshot_repo_commit
    plugin_name = "noop_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="noop",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
            version_hash="v1",
        )
    )
    service = _build_service(snapshot)
    cfg = GraphMetricsStepConfig(
        snapshot=snapshot,
        enabled_plugins=(plugin_name,),
        plugin_policy=GraphPluginPolicy(skip_on_unchanged=True),
    )
    manifest_path = tmp_path / "manifest.json"
    # First run writes manifest
    first_report = service.run_plugins(
        (plugin_name,),
        cfg=cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    if first_report.records[0].status != "succeeded":
        pytest.fail("Initial run should succeed")
    # Second run should skip based on unchanged input_hash
    second_report = service.run_plugins(
        (plugin_name,),
        cfg=cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    if second_report.records[0].status != "skipped":
        pytest.fail("Expected plugin to be skipped due to unchanged manifest")
    if second_report.records[0].skipped_reason != "unchanged":
        pytest.fail("Skipped reason should indicate unchanged input")


def test_dry_run_skips_execution(
    tmp_path: Path, snapshot_repo_commit: tuple[str, str, SnapshotRef]
) -> None:
    """Dry-run should not execute plugins and mark status skipped."""
    _, _, snapshot = snapshot_repo_commit
    plugin_name = "dry_run_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="noop",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: (_ for _ in ()).throw(RuntimeError("should not run")),
            version_hash="v1",
        )
    )
    service = _build_service(snapshot)
    cfg = GraphMetricsStepConfig(
        snapshot=snapshot,
        enabled_plugins=(plugin_name,),
        plugin_policy=GraphPluginPolicy(dry_run=True),
    )
    manifest_path = tmp_path / "manifest.json"
    report = service.run_plugins(
        (plugin_name,),
        cfg=cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    record = report.records[0]
    if record.status != "skipped":
        pytest.fail("Dry-run should mark plugin as skipped")
    if record.skipped_reason != "dry_run":
        pytest.fail("Skipped reason should indicate dry_run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("records", [])[0].get("status") != "skipped":
        pytest.fail("Manifest should record skipped status for dry run")


def test_manifest_records_contract_results(
    tmp_path: Path, snapshot_repo_commit: tuple[str, str, SnapshotRef]
) -> None:
    """Manifest should include contract result entries."""
    _, _, snapshot = snapshot_repo_commit
    plugin_name = "contract_manifest_plugin"
    checker = lambda _ctx: PluginContractResult(name="demo_contract", status="passed")  # noqa: E731
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="contract manifest",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
            contract_checkers=(checker,),
        )
    )
    service = _build_service(snapshot)
    cfg = GraphMetricsStepConfig(snapshot=snapshot, enabled_plugins=(plugin_name,))
    manifest_path = tmp_path / "manifest.json"
    try:
        report = service.run_plugins(
            (plugin_name,),
            cfg=cfg,
            run_options=GraphPluginRunOptions(manifest_path=manifest_path),
        )
    finally:
        unregister_graph_metric_plugin(plugin_name)
    record = report.records[0]
    if not record.contracts:
        pytest.fail("Expected contract results on record")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = manifest.get("records", [])[0].get("contracts", [])
    if not contracts:
        pytest.fail("Manifest should include contracts for plugin")
    if contracts[0].get("status") != "passed":
        pytest.fail("Contract status should be persisted to manifest")


def test_manifest_records_scope(
    tmp_path: Path, snapshot_repo_commit: tuple[str, str, SnapshotRef]
) -> None:
    """Manifest should include scopes used during execution."""
    _, _, snapshot = snapshot_repo_commit
    plugin_name = "scope_manifest_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="scope manifest",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
        )
    )
    scope = GraphRunScope(
        paths=("a.py",),
        modules=("mod.a",),
        time_window=(datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 2, tzinfo=UTC)),
    )
    cfg = GraphMetricsStepConfig(snapshot=snapshot, enabled_plugins=(plugin_name,), scope=scope)
    service = _build_service(snapshot)
    manifest_path = tmp_path / "manifest.json"
    try:
        service.run_plugins(
            (plugin_name,),
            cfg=cfg,
            run_options=GraphPluginRunOptions(manifest_path=manifest_path),
        )
    finally:
        unregister_graph_metric_plugin(plugin_name)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope_payload = payload.get("scope", {})
    if scope_payload.get("paths") != list(scope.paths):
        pytest.fail("Scope paths should be written to manifest")
    if scope_payload.get("modules") != list(scope.modules):
        pytest.fail("Scope modules should be written to manifest")
    time_window = scope_payload.get("time_window")
    expected_time_bounds = 2
    if not isinstance(time_window, (list, tuple)) or len(time_window) != expected_time_bounds:
        pytest.fail("Scope time_window should be written to manifest")
    if scope.time_window is None:
        pytest.fail("Expected scope time_window to be set on config")
    if time_window[0] != scope.time_window[0].isoformat():
        pytest.fail("Scope time_window should be written to manifest")
    if time_window[1] != scope.time_window[1].isoformat():
        pytest.fail("Scope time_window upper bound should be written to manifest")
