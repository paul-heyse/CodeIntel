"""Graph plugin executor.

This module provides the execution infrastructure for running graph plugins
without any dependency on the analytics subsystem.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from codeintel.config.steps_graphs import GraphPluginPolicy
from codeintel.core.plugins.context import PluginScratch
from codeintel.core.plugins.result import PluginExecutionRecord, PluginResult
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource
from codeintel.graphs.runtime.manifest import (
    ManifestState,
    RecordParams,
    dry_run_record,
    is_unchanged,
    skip_record,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    GraphPluginExecutionPlan,
    PluginExecutionSettings,
    plan_graph_plugin_run,
)
from codeintel.storage.db_helpers import DUCKDB_ERRORS
from codeintel.storage.run_tracking import PipelineStatus, PipelineStepRecord, StepStatus

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.graphs.engine import GraphEngine, NxGraphEngine
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.run_tracking import PipelineRunTracking

log = logging.getLogger(__name__)

# Errors that can be caught and handled during plugin execution
PLUGIN_CATCHABLE_ERRORS: tuple[type[Exception], ...] = (
    *DUCKDB_ERRORS,
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


class PluginFatalError(Exception):
    """Fatal plugin failure while respecting fail-fast semantics."""

    def __init__(self, record: PluginExecutionRecord, original: Exception) -> None:
        """Initialize with execution record and original exception.

        Parameters
        ----------
        record
            The execution record at time of failure.
        original
            The exception that caused the failure.
        """
        super().__init__(str(original))
        self.record = record


@dataclass(frozen=True)
class GraphRunReport:
    """Report of a graph plugin execution run.

    Attributes
    ----------
    run_id
        Unique run identifier.
    repo
        Repository identifier.
    commit
        Commit SHA.
    records
        Execution records for each plugin.
    success_count
        Number of successful executions.
    failure_count
        Number of failed executions.
    skip_count
        Number of skipped executions.
    duration_ms
        Total run duration in milliseconds.
    started_at
        Run start time.
    ended_at
        Run end time.
    fatal_error
        Whether run ended due to fatal error.
    manifest
        Final manifest state.
    """

    run_id: str
    repo: str
    commit: str
    records: tuple[PluginExecutionRecord, ...]
    success_count: int
    failure_count: int
    skip_count: int
    duration_ms: float
    started_at: datetime
    ended_at: datetime
    fatal_error: bool = False
    manifest: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass
class GraphExecutorContext:
    """Context for graph plugin execution.

    Attributes
    ----------
    gateway
        Storage gateway.
    snapshot
        Repository snapshot.
    engine
        Graph engine.
    catalog_provider
        Function catalog provider.
    run_context
        Optional unified run context for cross-engine correlation.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    engine: GraphEngine | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    run_context: RunContext | None = None


def _run_with_timeout(
    plugin: GraphPluginProtocol,
    ctx: GraphPluginExecutionContext,
    timeout_ms: int | None,
) -> PluginResult:
    """Execute a plugin with an optional timeout.

    Parameters
    ----------
    plugin
        Plugin to execute.
    ctx
        Execution context.
    timeout_ms
        Timeout in milliseconds.

    Returns
    -------
    PluginResult
        Plugin execution result.

    Raises
    ------
    TimeoutError
        If execution exceeds timeout.
    """
    if timeout_ms is None:
        return plugin.execute(ctx)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(plugin.execute, ctx)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FuturesTimeout as exc:
            future.cancel()
            message = f"Graph plugin timed out after {timeout_ms} ms"
            raise TimeoutError(message) from exc


def _execute_plugin(
    *,
    plugin: GraphPluginProtocol,
    ctx: GraphPluginExecutionContext,
    settings: PluginExecutionSettings,
) -> PluginExecutionRecord:
    """Execute a plugin with retry and timeout handling.

    Parameters
    ----------
    plugin
        Plugin to execute.
    ctx
        Execution context.
    settings
        Execution settings.

    Returns
    -------
    PluginExecutionRecord
        Execution record.

    Raises
    ------
    PluginFatalError
        When fatal failure occurs and fail-fast is enabled.
    """
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    attempts = 0
    status: Literal["succeeded", "failed", "skipped"] = "failed"
    error_message: str | None = None
    plugin_result: PluginResult | None = None
    max_attempts = max(settings.retry_cfg.max_attempts, 1)

    while attempts < max_attempts:
        attempts += 1
        try:
            plugin_result = _run_with_timeout(plugin, ctx, settings.timeout_ms)
            if plugin_result.success:
                status = "succeeded" if not plugin_result.skipped else "skipped"
                error_message = None
            else:
                status = "failed"
                error_message = plugin_result.error
            break
        except TimeoutError:
            error_message = "timeout"
            status = "failed"
            break
        except PLUGIN_CATCHABLE_ERRORS as exc:
            error_message = repr(exc)
            status = "skipped" if settings.severity == "skip_on_error" else "failed"
            if status == "failed" and attempts < max_attempts:
                _maybe_retry(plugin.metadata.name, attempts, max_attempts, settings)
                continue
            if status == "failed" and settings.severity == "fatal" and settings.fail_fast:
                record = PluginExecutionRecord(
                    plugin_name=plugin.metadata.name,
                    status=status,
                    started_at=started_at,
                    ended_at=datetime.now(tz=UTC),
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    attempts=attempts,
                    partial=True,
                    error=error_message,
                    meta={
                        "input_hash": settings.input_hash,
                        "options_hash": settings.options_hash,
                        "version_hash": settings.version_hash,
                    },
                )
                raise PluginFatalError(record, exc) from exc
            break

    ended_at = datetime.now(tz=UTC)
    row_counts = plugin_result.row_counts if plugin_result else None
    input_hash = (
        plugin_result.input_hash
        if plugin_result and plugin_result.input_hash
        else settings.input_hash
    )
    options_hash = (
        plugin_result.options_hash
        if plugin_result and plugin_result.options_hash
        else settings.options_hash
    )

    return PluginExecutionRecord(
        plugin_name=plugin.metadata.name,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        attempts=attempts,
        partial=status != "succeeded",
        error=error_message,
        meta={
            "row_counts": dict(row_counts) if row_counts else None,
            "input_hash": input_hash,
            "options_hash": options_hash,
            "version_hash": settings.version_hash,
            "severity": settings.severity,
        },
    )


def _maybe_retry(
    plugin_name: str,
    attempts: int,
    max_attempts: int,
    settings: PluginExecutionSettings,
) -> None:
    """Log and back off before a retry attempt.

    Parameters
    ----------
    plugin_name
        Name of the plugin being retried.
    attempts
        Current attempt number.
    max_attempts
        Maximum attempts allowed.
    settings
        Execution settings.
    """
    backoff_ms = settings.retry_cfg.backoff_ms or 0
    log.warning(
        "graph_runtime.plugin.retry name=%s attempt=%d/%d backoff_ms=%d",
        plugin_name,
        attempts,
        max_attempts,
        backoff_ms,
    )
    if backoff_ms > 0:
        time.sleep(backoff_ms / 1000)


def _execute_planned_plugin(
    *,
    plugin: GraphPluginProtocol,
    ctx: GraphPluginExecutionContext,
    settings: PluginExecutionSettings,
    plan: GraphPluginExecutionPlan,
) -> PluginExecutionRecord:
    """Execute a plugin according to the execution plan.

    Parameters
    ----------
    plugin
        Plugin to execute.
    ctx
        Execution context.
    settings
        Execution settings.
    plan
        Execution plan.

    Returns
    -------
    PluginExecutionRecord
        Execution record.

    Raises
    ------
    PluginFatalError
        When fatal failure occurs.
    RuntimeError
        When plugin execution fails to produce a record (defensive check).
    """
    span = plan.telemetry.start_plugin(plugin, plan.run_id, ctx)

    log.info(
        "graph_runtime.plugin.start name=%s repo=%s commit=%s stage=%s",
        plugin.metadata.name,
        ctx.repo,
        ctx.commit,
        plugin.metadata.stage,
        extra={"graph_run_id": plan.run_id},
    )

    params = RecordParams(
        severity=settings.severity,
        timeout_ms=settings.timeout_ms,
        version_hash=settings.version_hash,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        options=ctx.options,
        requires_isolation=plugin.metadata.isolation_kind != "none",
        isolation_kind=plugin.metadata.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )

    state = ManifestState(
        plugin_name=plugin.metadata.name,
        row_count_tables=plugin.metadata.row_count_tables,
        gateway=ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
    )

    record: PluginExecutionRecord | None = None
    try:
        if plan.policy.dry_run:
            record = dry_run_record(plugin=plugin, params=params)
        elif plan.policy.skip_on_unchanged and is_unchanged(plan.prior_manifest, state):
            record = skip_record(plugin=plugin, params=params, reason="unchanged")
        else:
            record = _execute_plugin(
                plugin=plugin,
                ctx=ctx,
                settings=settings,
            )
    except PluginFatalError:
        raise
    except PLUGIN_CATCHABLE_ERRORS:
        log.exception("plugin_failed", extra={"graph_run_id": plan.run_id})
        now = datetime.now(tz=UTC)
        record = PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status="failed",
            started_at=now,
            ended_at=now,
            duration_ms=0.0,
            attempts=1,
            partial=True,
            error="plugin_failed",
            meta={
                "input_hash": settings.input_hash,
                "options_hash": settings.options_hash,
                "version_hash": settings.version_hash,
            },
        )

    if record is None:  # pragma: no cover
        missing_record_message = "Plugin execution did not produce a record"
        raise RuntimeError(missing_record_message)

    plan.telemetry.finish_plugin(span, record)
    plan.telemetry.record_metrics(record, plan.scope)

    log.info(
        "graph_runtime.plugin.finish name=%s status=%s duration_ms=%.2f",
        record.plugin_name,
        record.status,
        record.duration_ms,
        extra={"graph_run_id": plan.run_id},
    )

    return record


def _execute_plugins_in_plan(
    *,
    plan: GraphPluginExecutionPlan,
    context: GraphExecutorContext,
) -> tuple[list[PluginExecutionRecord], dict[str, dict[str, object]], bool]:
    """
    Execute all plugins in a plan, returning records and manifest.

    Returns
    -------
    tuple[list[PluginExecutionRecord], dict[str, dict[str, object]], bool]
        Plugin records, manifest payload, and fatal_error flag.
    """
    records: list[PluginExecutionRecord] = []
    manifest: dict[str, dict[str, object]] = {}
    fatal_error = False
    scratch = PluginScratch()

    try:
        for plugin in plan.plugins:
            settings = plan.settings_by_plugin[plugin.metadata.name]
            options = plan.options_by_plugin.get(plugin.metadata.name)

            resources = ResourceContainer()
            resources.register(StorageResource(context.gateway, context.snapshot.repo_root))
            if context.engine is not None:
                resources.register(GraphResource(cast("NxGraphEngine", context.engine)))

            plugin_ctx = GraphPluginExecutionContext(
                gateway=context.gateway,
                snapshot=context.snapshot,
                run_id=plan.run_id,
                graph_resources=resources,
                scratch=scratch,
                options=options,
                plugin_name=plugin.metadata.name,
                scope=plan.scope,
                run_context=context.run_context,
                _catalog_provider=context.catalog_provider,
            )

            try:
                record = _execute_planned_plugin(
                    plugin=plugin,
                    ctx=plugin_ctx,
                    settings=settings,
                    plan=plan,
                )
            except PluginFatalError as exc:
                records.append(exc.record)
                fatal_error = True
                break

            records.append(record)

            if record.status == "succeeded":
                manifest[plugin.metadata.name] = {
                    "input_hash": record.meta.get("input_hash"),
                    "options_hash": record.meta.get("options_hash"),
                    "version_hash": record.meta.get("version_hash"),
                    "row_counts": record.meta.get("row_counts"),
                    "executed_at": record.ended_at,
                }
    finally:
        scratch.cleanup()

    return records, manifest, fatal_error


def _status_counts(records: Sequence[PluginExecutionRecord]) -> dict[str, int]:
    """
    Summarize plugin run statuses.

    Returns
    -------
    dict[str, int]
        Counts keyed by success/failure/skipped.
    """
    return {
        "success": sum(1 for r in records if r.status == "succeeded"),
        "failure": sum(1 for r in records if r.status == "failed"),
        "skipped": sum(1 for r in records if r.status == "skipped"),
    }


def run_graph_plugins(
    *,
    plan: GraphPluginExecutionPlan,
    context: GraphExecutorContext,
) -> GraphRunReport:
    """Execute all plugins in an execution plan.

    If a `run_context` is provided in the executor context, this function
    records run and step metadata to the pipeline registry.

    Returns
    -------
    GraphRunReport
        Summary of plugin execution outcomes.
    """
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)

    run_context = context.run_context
    runs = context.gateway.runs if run_context is not None else None
    if run_context is not None and runs is not None:
        runs.start_run(run_context, pipeline_name=f"graphs:{plan.scope}")

    run_span = plan.telemetry.start_run(
        run_id=plan.run_id,
        repo=plan.repo,
        commit=plan.commit,
        plugin_count=len(plan.plugins),
    )

    records, manifest, fatal_error = _execute_plugins_in_plan(plan=plan, context=context)

    ended_at = datetime.now(tz=UTC)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    status_counts = _status_counts(records)

    plan.telemetry.finish_run(
        run_span, status_counts["success"], status_counts["failure"], status_counts["skipped"]
    )

    if run_context is not None and runs is not None:
        _record_graph_steps(runs, run_context.run_id, records, plan)

        status: PipelineStatus
        error_summary: str | None = None
        if fatal_error or status_counts["failure"] > 0:
            status = "failed"
            failed_plugins = [r.plugin_name for r in records if r.status == "failed"]
            error_summary = f"Failed plugins: {', '.join(failed_plugins)}"
        elif status_counts["skipped"] > 0 and status_counts["success"] == 0:
            status = "partial"
        else:
            status = "succeeded"

        runs.complete_run(
            run_context.run_id,
            status=status,
            error_summary=error_summary,
        )

    return GraphRunReport(
        run_id=plan.run_id,
        repo=plan.repo,
        commit=plan.commit,
        records=tuple(records),
        success_count=status_counts["success"],
        failure_count=status_counts["failure"],
        skip_count=status_counts["skipped"],
        duration_ms=duration_ms,
        started_at=started_at,
        ended_at=ended_at,
        fatal_error=fatal_error,
        manifest=manifest,
    )


def _record_graph_steps(
    runs: PipelineRunTracking,
    run_id: str,
    records: list[PluginExecutionRecord],
    plan: GraphPluginExecutionPlan,
) -> None:
    """Record step records from graph results.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor from gateway.
    run_id
        Run identifier.
    records
        Graph plugin run records.
    plan
        Execution plan with plugin metadata.
    """
    for rec in records:
        # Get plugin metadata for stage information
        plugin_meta = next(
            (p.metadata for p in plan.plugins if p.metadata.name == rec.plugin_name),
            None,
        )
        stage = plugin_meta.stage if plugin_meta else "unknown"

        # Extract row_counts from meta if present
        row_counts_raw = rec.meta.get("row_counts")
        row_counts: dict[str, int] | None = None
        if isinstance(row_counts_raw, dict):
            row_counts = {str(k): int(v) for k, v in row_counts_raw.items()}

        # Map graph status to step status
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
                module="graphs",
                stage=stage,
                name=rec.plugin_name,
                status=step_status,
                started_at=rec.started_at,
                completed_at=rec.ended_at,
                row_counts=row_counts,
                extra=extra if extra else None,
            ),
        )


def run_graph_plugin_batch(
    *,
    plugins: Sequence[GraphPluginProtocol],
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    engine: GraphEngine | None = None,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> GraphRunReport:
    """Execute a batch of plugins with minimal configuration.

    Parameters
    ----------
    plugins
        Plugins to execute.
    gateway
        Storage gateway.
    snapshot
        Repository snapshot.
    engine
        Graph engine.
    catalog_provider
        Function catalog provider.

    Returns
    -------
    GraphRunReport
        Report of execution results.
    """
    context = GraphPlanContext(
        runtime_snapshot=snapshot,
        policy=GraphPluginPolicy(),
    )

    plan = plan_graph_plugin_run(
        plugin_names=[p.metadata.name for p in plugins],
        context=context,
    )

    executor_context = GraphExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        engine=engine,
        catalog_provider=catalog_provider,
    )

    return run_graph_plugins(plan=plan, context=executor_context)


__all__ = [
    "GraphExecutorContext",
    "GraphRunReport",
    "PluginFatalError",
    "run_graph_plugin_batch",
    "run_graph_plugins",
]
