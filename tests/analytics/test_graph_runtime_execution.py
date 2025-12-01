"""Unit tests for graph runtime execution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graphs.plugins import GraphMetricPlugin, GraphPluginResult
from codeintel.analytics.graphs.runtime.execution import BatchContext, run_graph_plugin_batch
from codeintel.analytics.graphs.runtime.manifest import InputHashPayload, compute_input_hash
from codeintel.analytics.graphs.runtime.model import GraphPluginRunOptions
from codeintel.analytics.graphs.runtime.planning import PlanContext, plan_graph_plugin_run
from codeintel.analytics.graphs.runtime.telemetry import NoOpGraphRuntimeTelemetry
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.plugin_packs import build_graph_plugin_pack


def _build_runtime(
    repo: str = "demo/repo",
    commit: str = "deadbeef",
) -> tuple[StorageGateway, GraphRuntime, GraphMetricsStepConfig, AnalyticsContext | None]:
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
    )
    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    return gateway, runtime, cfg, None


def test_run_graph_plugin_batch_executes_plugin() -> None:
    """run_graph_plugin_batch should execute plugins in the plan."""
    pack = build_graph_plugin_pack()
    plugin = pack.success
    pack.register(plugin)
    gateway, runtime, cfg, analytics_ctx = _build_runtime()
    try:
        plan = plan_graph_plugin_run(
            plugin_names=(plugin.name,),
            context=PlanContext(
                cfg=cfg,
                runtime_snapshot=cfg.snapshot,
                target=None,
                policy=GraphPluginPolicy(),
                run_options=GraphPluginRunOptions(),
                prior_manifest=None,
                telemetry=NoOpGraphRuntimeTelemetry(),
            ),
        )
        records = run_graph_plugin_batch(
            plan=plan,
            context=BatchContext(
                gateway=gateway,
                runtime=runtime,
                cfg=cfg,
                analytics_context=analytics_ctx,
                catalog_provider=None,
            ),
        )
    finally:
        pack.unregister_all()
    if len(records) != 1:
        pytest.fail("Expected a single record from execution batch")
    record = records[0]
    if record.status != "succeeded":
        pytest.fail("Plugin should succeed during execution")
    if record.run_id != plan.run_id:
        pytest.fail("Run ID should propagate to execution record")


def test_run_graph_plugin_batch_skips_when_unchanged() -> None:
    """Skip-on-unchanged should produce a skipped record."""
    pack = build_graph_plugin_pack()
    plugin = GraphMetricPlugin(
        name="skip_manifest_plugin",
        description="skip test",
        stage="core",
        enabled_by_default=False,
        run=lambda _ctx: GraphPluginResult(row_counts={"analytics.graph_metrics_functions": 0}),
        version_hash="v2",
    )
    pack.register(plugin)
    gateway, runtime, cfg, analytics_ctx = _build_runtime()
    scope = cfg.scope
    input_hash = compute_input_hash(
        InputHashPayload(
            repo=cfg.repo,
            commit=cfg.commit,
            plugin_name=plugin.name,
            version_hash=plugin.version_hash,
            scope=scope,
            options_hash=None,
        )
    )
    prior_manifest = {
        plugin.name: {
            "status": "succeeded",
            "input_hash": input_hash,
            "options_hash": None,
        }
    }
    try:
        plan = plan_graph_plugin_run(
            plugin_names=(plugin.name,),
            context=PlanContext(
                cfg=cfg,
                runtime_snapshot=cfg.snapshot,
                target=None,
                policy=GraphPluginPolicy(skip_on_unchanged=True),
                run_options=GraphPluginRunOptions(),
                prior_manifest=prior_manifest,
                telemetry=NoOpGraphRuntimeTelemetry(),
            ),
        )
        records = run_graph_plugin_batch(
            plan=plan,
            context=BatchContext(
                gateway=gateway,
                runtime=runtime,
                cfg=cfg,
                analytics_context=analytics_ctx,
                catalog_provider=None,
            ),
        )
    finally:
        pack.unregister_all()
    if len(records) != 1:
        pytest.fail("Expected a single record when skipping unchanged plugin")
    skipped = records[0]
    if skipped.status != "skipped" or skipped.skipped_reason != "unchanged":
        pytest.fail("Unchanged plugin should be skipped with reason 'unchanged'")
