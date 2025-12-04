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

from codeintel.analytics.core.context import (
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.analytics.core.executor import ExecutionPolicy, PluginExecutor
from codeintel.analytics.core.registry import PluginPlan, get_registry
from codeintel.analytics.plugins.middleware.logging import LoggingMiddleware
from codeintel.analytics.plugins.middleware.metrics import MetricsMiddleware
from codeintel.analytics.plugins.registration import ensure_plugins_registered
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.runtime import GraphRuntime
from codeintel.analytics.runtime.manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.tracking import PipelineStatus, PipelineStepRecord, StepStatus

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.runtime import RunContext
    from codeintel.storage.tracking import PipelineRunTracking

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
    graph_runtime: GraphRuntime | None
    cfgs: Mapping[str, Any]
    extra: Mapping[str, Any]
    catalog_provider: FunctionCatalogProvider | None = None
    snapshot: SnapshotRef | None = None


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
    unified_run_context: RunContext | None = None,
) -> PluginExecutionContextBuilder:
    """Build the execution context from plan and run context.

    Register resource providers for the new architecture. All plugins have
    been migrated to use ctx.require(ProviderType) pattern.

    Parameters
    ----------
    plan
        Analytics plugin execution plan.
    run_context
        Analytics run context with gateway and configs.
    snapshot
        Repository snapshot reference.
    unified_run_context
        Optional unified run context for cross-engine correlation.

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

    if unified_run_context is not None:
        builder = builder.with_run_context(unified_run_context)

    # Register resource providers (new architecture)
    if run_context.graph_runtime is not None:
        graph_provider = GraphProvider.from_runtime(run_context.graph_runtime)
        builder = builder.with_resource_provider(GraphProvider, graph_provider)

    if run_context.catalog_provider is not None:
        catalog_provider = CatalogProvider.from_catalog(run_context.catalog_provider)
        builder = builder.with_resource_provider(CatalogProvider, catalog_provider)

    # Register AST and features providers if snapshot is available
    if snapshot is not None:
        ast_provider = AstProvider(run_context.gateway, snapshot)
        builder = builder.with_resource_provider(AstProvider, ast_provider)
        features_provider = FeaturesProvider(run_context.gateway, snapshot)
        builder = builder.with_resource_provider(FeaturesProvider, features_provider)

    for config in run_context.cfgs.values():
        if config is not None:
            builder = builder.with_config(type(config), config)

    builder = builder.with_scope(
        AnalyticsScope(
            paths=plan.scope.paths,
            modules=plan.scope.modules,
            time_window=plan.scope.time_window,
            labels={"runtime": "analytics"},
        )
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


def run_analytics_plugins_for_context(
    *,
    unified_run_context: RunContext,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    enable_middleware: bool = True,
) -> AnalyticsRunReport:
    """Execute analytics plugins with unified RunContext.

    This is the preferred entrypoint that accepts a unified RunContext
    for consistent run identity across all engines. It records run and
    step metadata to the pipeline registry.

    Parameters
    ----------
    unified_run_context
        Unified run context for cross-engine correlation.
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

    Examples
    --------
    >>> from codeintel.runtime import new_run_context
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>> # Create unified context
    >>> snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> run_ctx = new_run_context(snapshot=snapshot, kind="analytics", trigger="cli")
    >>> # result = run_analytics_plugins_for_context(
    >>> #     unified_run_context=run_ctx, plan=plan, run_context=analytics_ctx
    >>> # )
    """
    runs = run_context.gateway.runs

    # Start the run in the registry
    runs.start_run(
        unified_run_context,
        pipeline_name=f"analytics:{plan.scope}",
    )

    snapshot = (
        unified_run_context.snapshot
        if run_context.snapshot is None
        else _extract_snapshot(plan, run_context)
    )
    builder = _build_execution_context(
        plan, run_context, snapshot, unified_run_context=unified_run_context
    )
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

    # Record steps from analytics records
    _record_analytics_steps(runs, unified_run_context.run_id, records)

    # Determine overall status
    status: PipelineStatus
    error_summary: str | None = None
    if any(r.status == "failed" for r in records):
        status = "failed"
        failed_plugins = [r.name for r in records if r.status == "failed"]
        error_summary = f"Failed plugins: {', '.join(failed_plugins)}"
    elif any(r.partial for r in records):
        status = "partial"
    else:
        status = "succeeded"

    # Complete the run
    runs.complete_run(
        unified_run_context.run_id,
        status=status,
        error_summary=error_summary,
    )

    analytics_scope = AnalyticsScope(
        paths=plan.scope.paths,
        modules=plan.scope.modules,
        time_window=plan.scope.time_window,
        labels={"runtime": "analytics"},
    )

    return AnalyticsRunReport(
        repo=plan.repo,
        commit=plan.commit,
        run_id=unified_run_context.run_id,
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


def _record_analytics_steps(
    runs: PipelineRunTracking,
    run_id: str,
    records: list[AnalyticsRunRecord],
) -> None:
    """Record step records from analytics results.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor from gateway.
    run_id
        Run identifier.
    records
        Analytics run records.
    """
    for rec in records:
        # Extract row_counts from meta if present
        meta = rec.meta or {}
        row_counts = meta.get("row_counts") if isinstance(meta.get("row_counts"), dict) else None

        # Map analytics status to step status
        step_status: StepStatus
        if rec.status == "succeeded":
            step_status = "succeeded"
        elif rec.status == "failed":
            step_status = "failed"
        elif rec.status == "skipped":
            step_status = "skipped"
        else:
            step_status = "failed"

        # Build extra metadata
        extra: dict[str, object] = {}
        if rec.error:
            extra["error"] = rec.error
        if rec.partial:
            extra["partial"] = True
        if rec.attempts > 1:
            extra["attempts"] = rec.attempts

        runs.record_step(
            PipelineStepRecord(
                run_id=run_id,
                module="analytics",
                stage=rec.kind,
                name=rec.name,
                status=step_status,
                started_at=rec.started_at,
                completed_at=rec.ended_at,
                row_counts=row_counts,
                extra=extra if extra else None,
            ),
        )


__all__ = [
    "AnalyticsPlanRequest",
    "AnalyticsPluginExecutionPlan",
    "AnalyticsRunContext",
    "plan_analytics_plugin_run",
    "run_analytics_plugins",
    "run_analytics_plugins_for_context",
]
