"""Bridge module for pipeline integration with the unified plugin system.

This module provides functions that allow the pipeline orchestration
steps to work with the unified analytics plugin system.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.context import AnalyticsContext, AnalyticsContextConfig
from codeintel.analytics.core.execution_context import (
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.analytics.core.executor import ExecutionPolicy, PluginExecutor
from codeintel.analytics.core.plugins.middleware.logging import LoggingMiddleware
from codeintel.analytics.core.plugins.middleware.metrics import MetricsMiddleware
from codeintel.analytics.core.plugins.registration import ensure_plugins_registered
from codeintel.analytics.core.registry import PluginPlan, get_registry
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.runtime_manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsPlanRequest:
    """Inputs required to plan an analytics plugin run."""

    plugin_names: Sequence[str]
    policy: GraphPluginPolicy
    repo: str
    commit: str
    scope: GraphRunScope
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    cfg_options: Mapping[str, dict[str, object]] | None
    runtime_options: Mapping[str, dict[str, object]] | None
    run_id: str
    telemetry: object | None = None


@dataclass(frozen=True)
class AnalyticsRunContext:
    """Shared run-time context for analytics plugin execution."""

    gateway: StorageGateway
    analytics_context: AnalyticsContext | None
    graph_runtime: GraphRuntime | None
    cfgs: Mapping[str, Any]
    extra: Mapping[str, Any]
    catalog_provider: FunctionCatalogProvider | None = None


@dataclass(frozen=True)
class AnalyticsPluginExecutionPlan:
    """Execution plan for a batch of analytics plugins."""

    plan_id: str
    run_id: str
    repo: str
    commit: str
    policy: GraphPluginPolicy
    scope: GraphRunScope
    ordered_names: tuple[str, ...]
    skipped: tuple[AnalyticsSkippedStep, ...]
    dep_graph: dict[str, tuple[str, ...]]
    internal_plan: PluginPlan


def plan_analytics_plugin_run(request: AnalyticsPlanRequest) -> AnalyticsPluginExecutionPlan:
    """Plan an analytics plugin execution run.

    Parameters
    ----------
    request
        Complete planning inputs including policy, scope, and plugin names.

    Returns
    -------
    AnalyticsPluginExecutionPlan
        Ordered plugin plan with resolved options and execution settings.
    """
    # Ensure plugins are registered
    ensure_plugins_registered()

    # Get the registry and create the plan
    registry = get_registry()
    internal_plan = registry.plan(
        plugin_names=list(request.plugin_names),
    )

    # Convert skipped to AnalyticsSkippedStep
    skipped = tuple(
        AnalyticsSkippedStep(
            name=skip.name,
            reason=skip.reason,
            kind="analytics_plugin",
        )
        for skip in internal_plan.skipped
    )

    return AnalyticsPluginExecutionPlan(
        plan_id=internal_plan.plan_id,
        run_id=request.run_id,
        repo=request.repo,
        commit=request.commit,
        policy=request.policy,
        scope=request.scope,
        ordered_names=internal_plan.ordered_names,
        skipped=skipped,
        dep_graph=internal_plan.dep_graph,
        internal_plan=internal_plan,
    )


def _extract_snapshot(
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
) -> SnapshotRef:
    """Extract snapshot from plan and configs.

    Returns
    -------
    SnapshotRef
        Snapshot reference with repo, commit, and repo_root.
    """
    repo_root = Path()
    for cfg in run_context.cfgs.values():
        if hasattr(cfg, "repo_root") and cfg.repo_root:
            repo_root = cfg.repo_root
            break
    return SnapshotRef(repo=plan.repo, commit=plan.commit, repo_root=repo_root)


def _build_execution_context(
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    snapshot: SnapshotRef,
) -> PluginExecutionContextBuilder:
    """Build the execution context from plan and run context.

    Register both legacy context objects and new resource providers for
    backward compatibility during migration.

    Returns
    -------
    PluginExecutionContextBuilder
        Builder configured with all context components.
    """
    builder = PluginExecutionContextBuilder(
        gateway=run_context.gateway,
        snapshot=snapshot,
        run_id=plan.run_id,
    )

    # Register resource providers (new architecture)
    if run_context.graph_runtime is not None:
        graph_provider = GraphProvider.from_runtime(run_context.graph_runtime)
        builder = builder.with_resource_provider(GraphProvider, graph_provider)
        # Also keep legacy for backward compat
        builder = builder.with_graph_runtime(run_context.graph_runtime)

    if run_context.catalog_provider is not None:
        catalog_provider = CatalogProvider.from_catalog(run_context.catalog_provider)
        builder = builder.with_resource_provider(CatalogProvider, catalog_provider)
        # Also keep legacy for backward compat
        builder = builder.with_catalog(run_context.catalog_provider)

    if run_context.analytics_context is not None:
        # Create provider from existing context
        context_config = AnalyticsContextConfig(
            repo=snapshot.repo,
            commit=snapshot.commit,
            repo_root=snapshot.repo_root,
        )
        context_provider = AnalyticsContextProvider(
            run_context.gateway,
            context_config,
        )
        # Pre-load with existing context
        context_provider._value = run_context.analytics_context
        context_provider._is_loaded = True
        builder = builder.with_resource_provider(AnalyticsContextProvider, context_provider)
        # Also keep legacy for backward compat
        builder = builder.with_analytics_context(run_context.analytics_context)

    for config in run_context.cfgs.values():
        if config is not None:
            builder = builder.with_config(type(config), config)

    builder.scope = AnalyticsScope(
        paths=plan.scope.paths,
        modules=plan.scope.modules,
        time_window=plan.scope.time_window,
        labels={"runtime": "analytics"},
    )

    for key, value in run_context.extra.items():
        builder = builder.with_extra(key, value)

    return builder


def run_analytics_plugins(
    *,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    enable_middleware: bool = True,
) -> AnalyticsRunReport:
    """Execute all plugins in `plan` using the new unified executor.

    Parameters
    ----------
    plan
        Planned plugins with settings and dependency ordering.
    run_context
        Shared runtime context (gateway, runtimes, configs) for execution.
    enable_middleware
        Whether to enable logging and metrics middleware (default: True).

    Returns
    -------
    AnalyticsRunReport
        Telemetry and records for each executed plugin.
    """
    snapshot = _extract_snapshot(plan, run_context)
    builder = _build_execution_context(plan, run_context, snapshot)
    ctx = builder.build()

    policy = ExecutionPolicy(
        fail_fast=plan.policy.fail_fast,
        max_retries=0,
        skip_on_unchanged=plan.policy.skip_on_unchanged,
        dry_run=plan.policy.dry_run,
        validate_contracts=True,
    )

    # Configure middleware
    middleware = []
    if enable_middleware:
        middleware = [LoggingMiddleware(), MetricsMiddleware()]

    executor = PluginExecutor(policy=policy, middleware=middleware)
    scratch = PluginScratch()
    report = executor.execute(ctx, plan.internal_plan, scratch=scratch)

    records = _convert_execution_records(report.records)
    analytics_scope = AnalyticsScope(
        paths=plan.scope.paths,
        modules=plan.scope.modules,
        time_window=plan.scope.time_window,
        labels={"runtime": "analytics"},
    )

    return AnalyticsRunReport(
        repo=plan.repo,
        commit=plan.commit,
        run_id=plan.run_id,
        scope=analytics_scope,
        records=tuple(records),
        plan=AnalyticsPlanInfo(
            plan_id=plan.plan_id,
            ordered_steps=plan.ordered_names,
            skipped_steps=plan.skipped,
            dep_graph=plan.dep_graph,
        ),
        tags={"runtime": "analytics"},
    )


def _convert_execution_records(
    records: tuple[Any, ...],
) -> list[AnalyticsRunRecord]:
    """Convert execution records to analytics format.

    Returns
    -------
    list[AnalyticsRunRecord]
        Converted analytics run records.
    """
    result: list[AnalyticsRunRecord] = []
    for record in records:
        meta: dict[str, object] = {}
        if record.result is not None:
            meta["row_counts"] = record.result.row_counts
            meta["result"] = record.result.meta
        if record.error:
            meta["error"] = record.error

        result.append(
            AnalyticsRunRecord(
                name=record.plugin_name,
                kind="analytics_plugin",
                status=record.status,
                started_at=record.started_at,
                ended_at=record.ended_at,
                duration_ms=record.duration_ms,
                attempts=record.attempts,
                partial=record.status != "succeeded",
                error=record.error,
                meta=meta,
            )
        )
    return result


__all__ = [
    "AnalyticsPlanRequest",
    "AnalyticsPluginExecutionPlan",
    "AnalyticsRunContext",
    "plan_analytics_plugin_run",
    "run_analytics_plugins",
]
