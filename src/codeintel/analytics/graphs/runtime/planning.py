"""Planning utilities for graph plugin execution."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from codeintel.analytics.graphs.contracts import ContractChecker
from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    GraphMetricPluginSkip,
    plan_graph_metric_plugins,
    resolve_plugin_options,
)
from codeintel.analytics.graphs.runtime.manifest import (
    InputHashPayload,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.analytics.graphs.runtime.model import GraphPluginRunOptions
from codeintel.analytics.graphs.runtime.telemetry import GraphRuntimeTelemetry
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


@dataclass(frozen=True)
class PluginExecutionSettings:
    """Resolved execution policy and hashes for a single plugin."""

    name: str
    severity: PluginSeverity
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None
    contract_checkers: tuple[ContractChecker, ...]


@dataclass(frozen=True)
class PluginExecutionPlan:
    """Execution plan for a batch of graph metric plugins."""

    plan_id: str
    run_id: str
    repo: str
    commit: str
    policy: GraphPluginPolicy
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    telemetry: GraphRuntimeTelemetry
    scope: GraphRunScope
    plugins: tuple[GraphMetricPlugin, ...]
    ordered_names: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]
    settings_by_plugin: dict[str, PluginExecutionSettings]
    options_by_plugin: dict[str, object | None]


@dataclass(frozen=True)
class PlanContext:
    """Aggregate inputs required for planning a plugin run."""

    cfg: GraphMetricsStepConfig | None
    runtime_snapshot: SnapshotRef | None
    target: tuple[str, str] | None
    policy: GraphPluginPolicy
    run_options: GraphPluginRunOptions | None
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    telemetry: GraphRuntimeTelemetry


def _resolve_plugin_options_map(
    plugins: Sequence[GraphMetricPlugin],
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
) -> dict[str, object | None]:
    """
    Merge and validate plugin options from config and runtime overrides.

    Returns
    -------
    dict[str, object | None]
        Normalized options per plugin.

    Raises
    ------
    ValueError
        If options are supplied for unknown plugins.
    """
    cfg_options = cfg_options or {}
    runtime_options = runtime_options or {}
    allowed_plugins = {plugin.name for plugin in plugins}
    unknown_option_plugins = (set(cfg_options.keys()) | set(runtime_options.keys())) - allowed_plugins
    if unknown_option_plugins:
        message = (
            "Options provided for unknown graph metric plugins: "
            f"{', '.join(sorted(unknown_option_plugins))}"
        )
        raise ValueError(message)
    resolved: dict[str, object | None] = {}
    for plugin in plugins:
        resolved[plugin.name] = resolve_plugin_options(
            plugin,
            cfg_options.get(plugin.name),
            runtime_options.get(plugin.name),
        )
    return resolved


def _effective_severity(
    plugin: GraphMetricPlugin,
    policy: GraphPluginPolicy,
) -> PluginSeverity:
    """
    Resolve effective severity using policy overrides and plugin defaults.

    Returns
    -------
    PluginSeverity
        Severity applied to the plugin.
    """
    override = policy.severity_overrides.get(plugin.name)
    if override is not None:
        return override
    return policy.default_severity


def _effective_timeout(plugin: GraphMetricPlugin, policy: GraphPluginPolicy) -> int | None:
    """
    Resolve effective timeout using policy overrides or plugin hints.

    Returns
    -------
    int | None
        Timeout in milliseconds when available.
    """
    override = policy.timeouts_ms.get(plugin.name)
    if override is not None:
        return override
    hints = getattr(plugin, "resource_hints", None)
    return hints.max_runtime_ms if hints is not None else None


def _resolve_target(
    *,
    cfg: GraphMetricsStepConfig | None,
    runtime_snapshot: SnapshotRef | None,
    target: tuple[str, str] | None,
) -> tuple[str, str]:
    """
    Resolve repo/commit from config, explicit target, or runtime snapshot.

    Returns
    -------
    tuple[str, str]
        Resolved repository and commit identifiers.

    Raises
    ------
    ValueError
        If no target or snapshot is available.
    """
    if cfg is not None:
        return cfg.repo, cfg.commit
    if target is not None:
        return target
    if runtime_snapshot is None:
        message = "Graph runtime missing snapshot; cannot derive repo/commit"
        raise ValueError(message)
    return runtime_snapshot.repo, runtime_snapshot.commit


def plan_graph_plugin_run(
    *,
    plugin_names: Sequence[str],
    context: PlanContext,
) -> PluginExecutionPlan:
    """
    Build an execution plan and per-plugin settings for a batch run.

    Returns
    -------
    PluginExecutionPlan
        Concrete plan describing ordering, options, and hashes.
    """
    run_id = uuid.uuid4().hex
    plan_id = uuid.uuid4().hex
    cfg = context.cfg
    run_options = context.run_options
    policy = context.policy
    scope = run_options.scope if run_options and run_options.scope is not None else (
        cfg.scope if cfg is not None else GraphRunScope()
    )
    repo, commit = _resolve_target(
        cfg=cfg,
        runtime_snapshot=context.runtime_snapshot,
        target=context.target,
    )
    plan = plan_graph_metric_plugins(
        plugin_names,
        enabled=cfg.enabled_plugins if cfg is not None else None,
        disabled=cfg.disabled_plugins if cfg is not None else None,
        defaults=None,
    )
    plugins: tuple[GraphMetricPlugin, ...] = plan.plugins
    options_by_plugin = _resolve_plugin_options_map(
        plugins=plugins,
        cfg_options=cfg.plugin_options if cfg is not None else {},
        runtime_options=(run_options.plugin_options if run_options is not None else {}) or {},
    )
    settings_by_plugin: dict[str, PluginExecutionSettings] = {}
    for plugin in plugins:
        options = options_by_plugin.get(plugin.name)
        options_hash = compute_options_hash(plugin, options)
        input_hash = compute_input_hash(
            InputHashPayload(
                repo=repo,
                commit=commit,
                plugin_name=plugin.name,
                version_hash=plugin.version_hash,
                scope=scope,
                options_hash=options_hash,
            )
        )
        settings_by_plugin[plugin.name] = PluginExecutionSettings(
            name=plugin.name,
            severity=_effective_severity(plugin, policy),
            retry_cfg=policy.retries.get(plugin.name, GraphPluginRetryPolicy()),
            timeout_ms=_effective_timeout(plugin, policy),
            fail_fast=policy.fail_fast,
            input_hash=input_hash,
            options_hash=options_hash,
            version_hash=plugin.version_hash,
            contract_checkers=plugin.contract_checkers,
        )

    return PluginExecutionPlan(
        plan_id=plan_id,
        run_id=run_id,
        repo=repo,
        commit=commit,
        policy=policy,
        prior_manifest=context.prior_manifest,
        telemetry=context.telemetry,
        scope=scope,
        plugins=plugins,
        ordered_names=plan.ordered_names,
        skipped_plugins=plan.skipped_plugins,
        dep_graph=plan.dep_graph,
        settings_by_plugin=settings_by_plugin,
        options_by_plugin=dict(options_by_plugin),
    )


__all__ = [
    "PluginExecutionPlan",
    "PluginExecutionSettings",
    "plan_graph_plugin_run",
]
