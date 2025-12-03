"""Planning utilities for graph plugin execution.

This module provides execution planning for graph plugins without any
dependency on the analytics subsystem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.graphs.core.protocol import (
    DEFAULT_GRAPH_PLUGINS,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginSkip,
)
from codeintel.graphs.core.registry import get_graph_registry
from codeintel.graphs.runtime.manifest import (
    InputHashPayload,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.graphs.runtime.telemetry import GraphRuntimeTelemetry, get_graph_telemetry

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.config.steps_graphs import GraphMetricsStepConfig

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


@dataclass(frozen=True)
class PluginExecutionSettings:
    """Resolved execution policy and hashes for a single plugin.

    Attributes
    ----------
    name
        Plugin name.
    severity
        Failure severity level.
    retry_cfg
        Retry configuration.
    timeout_ms
        Execution timeout in milliseconds.
    fail_fast
        Whether to abort on first failure.
    input_hash
        Content hash of inputs.
    options_hash
        Content hash of options.
    version_hash
        Plugin version hash.
    """

    name: str
    severity: PluginSeverity
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None


@dataclass(frozen=True)
class GraphPluginExecutionPlan:
    """Execution plan for a batch of graph plugins.

    Attributes
    ----------
    plan_id
        Unique plan identifier.
    run_id
        Run identifier.
    repo
        Repository identifier.
    commit
        Commit SHA.
    policy
        Execution policy.
    prior_manifest
        Prior execution manifest for skip detection.
    telemetry
        Telemetry manager.
    scope
        Execution scope.
    plugins
        Ordered plugins to execute.
    ordered_names
        Plugin names in execution order.
    skipped_plugins
        Plugins that will be skipped.
    dep_graph
        Dependency graph.
    settings_by_plugin
        Per-plugin settings.
    options_by_plugin
        Per-plugin options.
    """

    plan_id: str
    run_id: str
    repo: str
    commit: str
    policy: GraphPluginPolicy
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    telemetry: GraphRuntimeTelemetry
    scope: GraphRunScope
    plugins: tuple[GraphPluginProtocol, ...]
    ordered_names: tuple[str, ...]
    skipped_plugins: tuple[GraphPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]
    settings_by_plugin: dict[str, PluginExecutionSettings]
    options_by_plugin: dict[str, object | None]


@dataclass(frozen=True)
class GraphPlanContext:
    """Inputs required for planning a plugin run.

    Attributes
    ----------
    cfg
        Graph metrics step configuration.
    runtime_snapshot
        Runtime snapshot reference.
    target
        Explicit target (repo, commit) tuple.
    policy
        Execution policy.
    run_options
        Runtime options.
    prior_manifest
        Prior manifest for skip detection.
    telemetry
        Telemetry manager.
    """

    cfg: GraphMetricsStepConfig | None = None
    runtime_snapshot: SnapshotRef | None = None
    target: tuple[str, str] | None = None
    policy: GraphPluginPolicy = field(default_factory=GraphPluginPolicy)
    run_options: GraphPluginRunOptions | None = None
    prior_manifest: Mapping[str, Mapping[str, object]] | None = None
    telemetry: GraphRuntimeTelemetry | None = None


@dataclass(frozen=True)
class GraphPluginRunOptions:
    """Runtime options for a plugin run.

    Attributes
    ----------
    scope
        Execution scope override.
    plugin_options
        Per-plugin options override.
    """

    scope: GraphRunScope | None = None
    plugin_options: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class ResolvedPlanInputs:
    """Resolved inputs for building a GraphPluginExecutionPlan."""

    plan_id: str
    run_id: str
    policy: GraphPluginPolicy
    telemetry: GraphRuntimeTelemetry
    scope: GraphRunScope
    repo: str
    commit: str
    plugin_plan: GraphPluginPlan
    options_by_plugin: dict[str, object | None]
    settings_by_plugin: dict[str, PluginExecutionSettings]
    prior_manifest: Mapping[str, Mapping[str, object]] | None


def _resolve_plugin_options_map(
    plugins: Sequence[GraphPluginProtocol],
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
) -> dict[str, object | None]:
    """Merge and validate plugin options from config and runtime overrides.

    Parameters
    ----------
    plugins
        Plugins to resolve options for.
    cfg_options
        Options from configuration.
    runtime_options
        Options from runtime overrides.

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
    allowed_plugins = {plugin.metadata.name for plugin in plugins}
    unknown_option_plugins = (
        set(cfg_options.keys()) | set(runtime_options.keys())
    ) - allowed_plugins
    if unknown_option_plugins:
        message = (
            "Options provided for unknown graph plugins: "
            f"{', '.join(sorted(unknown_option_plugins))}"
        )
        raise ValueError(message)

    resolved: dict[str, object | None] = {}
    for plugin in plugins:
        name = plugin.metadata.name
        cfg_opts = cfg_options.get(name)
        rt_opts = runtime_options.get(name)
        if cfg_opts is None and rt_opts is None:
            resolved[name] = plugin.metadata.options_default
        elif rt_opts is not None:
            # Runtime options override config
            resolved[name] = rt_opts
        else:
            resolved[name] = cfg_opts
    return resolved


def _effective_severity(
    plugin: GraphPluginProtocol,
    policy: GraphPluginPolicy,
) -> PluginSeverity:
    """Resolve effective severity using policy overrides and plugin defaults.

    Parameters
    ----------
    plugin
        Plugin to resolve severity for.
    policy
        Execution policy.

    Returns
    -------
    PluginSeverity
        Effective severity level.
    """
    override = policy.severity_overrides.get(plugin.metadata.name)
    if override is not None:
        return override
    return policy.default_severity


def _effective_timeout(plugin: GraphPluginProtocol, policy: GraphPluginPolicy) -> int | None:
    """Resolve effective timeout using policy overrides or plugin hints.

    Parameters
    ----------
    plugin
        Plugin to resolve timeout for.
    policy
        Execution policy.

    Returns
    -------
    int | None
        Timeout in milliseconds.
    """
    override = policy.timeouts_ms.get(plugin.metadata.name)
    if override is not None:
        return override
    hints = plugin.metadata.resource_hints
    return hints.max_runtime_ms if hints is not None else None


def _resolve_target(
    *,
    cfg: GraphMetricsStepConfig | None,
    runtime_snapshot: SnapshotRef | None,
    target: tuple[str, str] | None,
) -> tuple[str, str]:
    """Resolve repo/commit from config, explicit target, or runtime snapshot.

    Parameters
    ----------
    cfg
        Configuration.
    runtime_snapshot
        Runtime snapshot.
    target
        Explicit target.

    Returns
    -------
    tuple[str, str]
        Repository and commit identifiers.

    Raises
    ------
    ValueError
        If no target is available.
    """
    if cfg is not None:
        return cfg.repo, cfg.commit
    if target is not None:
        return target
    if runtime_snapshot is None:
        message = "Graph runtime missing snapshot; cannot derive repo/commit"
        raise ValueError(message)
    return runtime_snapshot.repo, runtime_snapshot.commit


def _prepare_execution_inputs(
    plugin_names: Sequence[str] | None,
    context: GraphPlanContext,
) -> ResolvedPlanInputs:
    """Resolve all derived inputs needed to build an execution plan."""
    run_id = uuid4().hex
    plan_id = uuid4().hex
    cfg = context.cfg
    run_options = context.run_options
    policy = context.policy
    telemetry = context.telemetry or get_graph_telemetry()

    scope = (
        run_options.scope
        if run_options and run_options.scope is not None
        else (cfg.scope if cfg is not None else GraphRunScope())
    )

    repo, commit = _resolve_target(
        cfg=cfg,
        runtime_snapshot=context.runtime_snapshot,
        target=context.target,
    )

    registry = get_graph_registry()
    enabled = cfg.enabled_plugins if cfg is not None else None
    disabled = cfg.disabled_plugins if cfg is not None else None
    plugin_plan: GraphPluginPlan = registry.plan(
        plugin_names=plugin_names or list(DEFAULT_GRAPH_PLUGINS),
        enabled=enabled,
        disabled=disabled,
        defaults=list(DEFAULT_GRAPH_PLUGINS),
    )

    plugins = plugin_plan.plugins
    options_by_plugin = _resolve_plugin_options_map(
        plugins=plugins,
        cfg_options=cfg.plugin_options if cfg is not None else {},
        runtime_options=(run_options.plugin_options if run_options is not None else {}) or {},
    )

    settings_by_plugin: dict[str, PluginExecutionSettings] = {}
    for plugin in plugins:
        name = plugin.metadata.name
        options = options_by_plugin.get(name)
        options_hash = compute_options_hash(plugin, options)
        input_hash = compute_input_hash(
            InputHashPayload(
                repo=repo,
                commit=commit,
                plugin_name=name,
                version_hash=plugin.metadata.version_hash,
                scope=scope,
                options_hash=options_hash,
            )
        )
        settings_by_plugin[name] = PluginExecutionSettings(
            name=name,
            severity=_effective_severity(plugin, policy),
            retry_cfg=policy.retries.get(name, GraphPluginRetryPolicy()),
            timeout_ms=_effective_timeout(plugin, policy),
            fail_fast=policy.fail_fast,
            input_hash=input_hash,
            options_hash=options_hash,
            version_hash=plugin.metadata.version_hash,
        )

    return ResolvedPlanInputs(
        plan_id=plan_id,
        run_id=run_id,
        policy=policy,
        telemetry=telemetry,
        scope=scope,
        repo=repo,
        commit=commit,
        plugin_plan=plugin_plan,
        options_by_plugin=dict(options_by_plugin),
        settings_by_plugin=settings_by_plugin,
        prior_manifest=context.prior_manifest,
    )


def plan_graph_plugin_run(
    *,
    plugin_names: Sequence[str] | None = None,
    context: GraphPlanContext,
) -> GraphPluginExecutionPlan:
    """Build an execution plan and per-plugin settings for a batch run.

    Parameters
    ----------
    plugin_names
        Explicit plugin names to run.
    context
        Planning context.

    Returns
    -------
    GraphPluginExecutionPlan
        Concrete plan for execution.
    """
    resolved = _prepare_execution_inputs(plugin_names, context)
    plugin_plan = resolved.plugin_plan
    return GraphPluginExecutionPlan(
        plan_id=resolved.plan_id,
        run_id=resolved.run_id,
        repo=resolved.repo,
        commit=resolved.commit,
        policy=resolved.policy,
        prior_manifest=resolved.prior_manifest,
        telemetry=resolved.telemetry,
        scope=resolved.scope,
        plugins=plugin_plan.plugins,
        ordered_names=plugin_plan.ordered_names,
        skipped_plugins=plugin_plan.skipped_plugins,
        dep_graph=plugin_plan.dep_graph,
        settings_by_plugin=resolved.settings_by_plugin,
        options_by_plugin=resolved.options_by_plugin,
    )


__all__ = [
    "GraphPlanContext",
    "GraphPluginExecutionPlan",
    "GraphPluginRunOptions",
    "PluginExecutionSettings",
    "plan_graph_plugin_run",
]
