"""Unit tests for graph metric plugin options and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import (
    GraphPluginRunOptions,
    GraphServiceRuntime,
)
from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    register_graph_metric_plugin,
    resolve_plugin_options,
    unregister_graph_metric_plugin,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from codeintel.storage.gateway import open_memory_gateway


class _OptionsModel(BaseModel):
    threshold: int = 1
    flag: bool = False


EXPECTED_THRESHOLD = 7


def _make_service() -> tuple[GraphServiceRuntime, GraphMetricsStepConfig]:
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
    )
    return GraphServiceRuntime(gateway=gateway, runtime=runtime), GraphMetricsStepConfig(
        snapshot=snapshot
    )


def test_options_defaults_merge_with_config_and_runtime_overrides() -> None:
    """Options merge order: default -> config -> runtime, validated via model."""
    seen: list[_OptionsModel] = []

    def _run(ctx: object) -> None:
        seen.append(ctx.options)  # type: ignore[attr-defined]

    plugin_name = "options_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="options test",
            stage="core",
            enabled_by_default=False,
            run=_run,
            options_model=_OptionsModel,
            options_default={"threshold": 2, "flag": False},
        )
    )
    service, cfg = _make_service()
    cfg = GraphMetricsStepConfig(
        snapshot=cfg.snapshot,
        plugin_options={plugin_name: {"flag": True}},
    )
    run_opts = GraphPluginRunOptions(
        plugin_options={plugin_name: {"threshold": EXPECTED_THRESHOLD}}
    )
    try:
        service.run_plugins((plugin_name,), cfg=cfg, run_options=run_opts)
    finally:
        unregister_graph_metric_plugin(plugin_name)
    if len(seen) != 1:
        pytest.fail("Plugin should have been invoked exactly once")
    options = seen[0]
    if not isinstance(options, _OptionsModel):
        pytest.fail("Plugin options should be validated into the declared model")
    if options.threshold != EXPECTED_THRESHOLD or options.flag is not True:
        pytest.fail("Options should merge default->config->runtime with runtime taking precedence")


def test_options_validation_error_surfaces() -> None:
    """Invalid options should raise during resolution before execution."""
    plugin_name = "options_validation_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="options validation",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
            options_model=_OptionsModel,
        )
    )
    service, cfg = _make_service()
    cfg = GraphMetricsStepConfig(
        snapshot=cfg.snapshot,
        plugin_options={plugin_name: {"threshold": "oops"}},  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(ValidationError):
            service.run_plugins((plugin_name,), cfg=cfg)
    finally:
        unregister_graph_metric_plugin(plugin_name)


def test_unknown_plugin_option_rejected() -> None:
    """Providing options for a plugin not in the plan should fail fast."""
    plugin_name = "options_unknown_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="known plugin",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
        )
    )
    service, cfg = _make_service()
    cfg = GraphMetricsStepConfig(
        snapshot=cfg.snapshot,
        plugin_options={"not_registered": {"foo": "bar"}},
    )
    try:
        with pytest.raises(ValueError, match="Options provided for unknown graph metric plugins"):
            service.run_plugins((plugin_name,), cfg=cfg)
    finally:
        unregister_graph_metric_plugin(plugin_name)


def test_resolve_plugin_options_merges_defaults_when_model_absent() -> None:
    """resolve_plugin_options should handle dict defaults without a model."""
    plugin = GraphMetricPlugin(
        name="options_no_model",
        description="no model plugin",
        stage="core",
        enabled_by_default=False,
        run=lambda _ctx: None,
        options_default={"a": 1},
    )
    resolved = resolve_plugin_options(plugin, {"b": 2}, {"a": 3})
    if resolved != {"a": 3, "b": 2}:
        pytest.fail("Options should merge defaults and overrides when no model is provided")
