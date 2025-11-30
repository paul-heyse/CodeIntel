"""Unit tests for graph runtime planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)
from codeintel.analytics.graphs.runtime.model import GraphPluginRunOptions
from codeintel.analytics.graphs.runtime.planning import PlanContext, plan_graph_plugin_run
from codeintel.analytics.graphs.runtime.telemetry import NoOpGraphRuntimeTelemetry
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy, GraphRunScope


def test_plan_graph_plugin_run_resolves_scope_and_hashes() -> None:
    """Planning should honor scope and produce input/options hashes."""
    plugin = GraphMetricPlugin(
        name="planning_demo",
        description="demo plugin",
        stage="core",
        enabled_by_default=False,
        run=lambda _ctx: None,
        version_hash="v1",
        options_default={"flag": True},
    )
    register_graph_metric_plugin(plugin)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    cfg = GraphMetricsStepConfig(
        snapshot=snapshot,
        plugin_options={plugin.name: {"threshold": 5}},
    )
    scope = GraphRunScope(paths=("src/demo.py",))
    try:
        plan = plan_graph_plugin_run(
            plugin_names=(plugin.name,),
            context=PlanContext(
                cfg=cfg,
                runtime_snapshot=snapshot,
                target=None,
                policy=GraphPluginPolicy(default_severity="soft_fail"),
                run_options=GraphPluginRunOptions(scope=scope),
                prior_manifest=None,
                telemetry=NoOpGraphRuntimeTelemetry(),
            ),
        )
    finally:
        unregister_graph_metric_plugin(plugin.name)
    if plan.scope.paths != scope.paths:
        pytest.fail("Scope paths should propagate to plan")
    if plan.repo != snapshot.repo or plan.commit != snapshot.commit:
        pytest.fail("Plan should inherit repo/commit from snapshot")
    settings = plan.settings_by_plugin[plugin.name]
    if settings.input_hash is None:
        pytest.fail("Input hash should be computed")
    if settings.options_hash is None:
        pytest.fail("Options hash should be computed")
    if settings.severity != "soft_fail":
        pytest.fail("Severity should honor policy default")
    if plan.options_by_plugin[plugin.name] is None:
        pytest.fail("Options should resolve from defaults/config/runtime")
