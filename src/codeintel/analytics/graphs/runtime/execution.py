"""Execution primitives for running graph metric plugins."""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any, Literal, cast

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    GraphRuntimeScratch,
    get_graph_metric_plugin,
    plan_graph_metric_plugins,
    register_graph_metric_plugin,
)
from codeintel.analytics.graphs.runtime.manifest import (
    ManifestState,
    RecordParams,
    dry_run_record,
    is_unchanged,
    run_contracts,
    skip_record,
)
from codeintel.analytics.graphs.runtime.model import GraphPluginRunRecord
from codeintel.analytics.graphs.runtime.planning import (
    PluginExecutionPlan,
    PluginExecutionSettings,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphRunScope
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    open_gateway,
    open_memory_gateway,
)

if TYPE_CHECKING:
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
else:  # pragma: no cover - fallback when dependency is absent at runtime

    class FunctionCatalogProvider:  # type: ignore[too-many-ancestors]
        """Placeholder to satisfy type checkers when catalog provider is unavailable."""


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IsolationEnvelope:
    """Serialized inputs for isolated plugin execution."""

    plugin_name: str
    plugin: GraphMetricPlugin | None
    repo: str
    commit: str
    options: object | None
    scope: GraphRunScope
    gateway_config: StorageConfig | None
    run_id: str


@dataclass(frozen=True)
class IsolationResult:
    """Serialized outputs from isolated plugin execution."""

    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    contracts: tuple[PluginContractResult, ...]
    row_counts: dict[str, int] | None = None
    input_hash: str | None = None
    options_hash: str | None = None


@dataclass(frozen=True)
class MPContext:
    """Minimal multiprocessing context for isolation support."""

    process_factory: Callable[..., multiprocessing.Process]
    queue_factory: Callable[[], multiprocessing.Queue[IsolationResult]]
    start_method_name: str

    def process(
        self,
        target: Callable[..., object],
        args: tuple[object, ...],
    ) -> multiprocessing.Process:
        """
        Create a multiprocessing.Process using the configured start method.

        Returns
        -------
        multiprocessing.Process
            Process configured with the desired start method.
        """
        return self.process_factory(target=target, args=args)

    def queue(self) -> multiprocessing.Queue[IsolationResult]:
        """
        Return a multiprocessing queue suitable for isolation results.

        Returns
        -------
        multiprocessing.Queue[IsolationResult]
            Queue bound to the multiprocessing context.
        """
        return self.queue_factory()

    def start_method(self) -> str:
        """
        Return the name of the start method (e.g., 'fork', 'spawn').

        Returns
        -------
        str
            Name of the multiprocessing start method.
        """
        return self.start_method_name


class PluginFatalError(Exception):
    """Fatal plugin failure while respecting fail-fast semantics."""

    def __init__(self, record: GraphPluginRunRecord, original: Exception) -> None:
        super().__init__(str(original))
        self.record = record


@dataclass(frozen=True)
class BatchContext:
    """Context for executing a planned plugin batch."""

    gateway: StorageGateway
    runtime: GraphRuntime
    cfg: GraphMetricsStepConfig | None
    analytics_context: AnalyticsContext | None
    catalog_provider: FunctionCatalogProvider | None


def _select_mp_context() -> MPContext:
    """
    Prefer forked processes to preserve in-memory plugin registration.

    Falls back to the default start method when fork is unavailable (e.g., Windows).

    Returns
    -------
    MPContext
        Multiprocessing context using the preferred start method.
    """
    base_ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )
    base_ctx_any = cast("Any", base_ctx)
    process_factory = cast(
        "Callable[..., multiprocessing.Process]",
        base_ctx_any.Process,
    )
    queue_factory = cast(
        "Callable[[], multiprocessing.Queue[IsolationResult]]",
        base_ctx_any.Queue,
    )
    return MPContext(
        process_factory=process_factory,
        queue_factory=queue_factory,
        start_method_name=base_ctx.get_start_method(),
    )


def _build_gateway_for_isolation(config: StorageConfig | None) -> StorageGateway:
    """
    Open a storage gateway for isolated execution.

    Returns
    -------
    StorageGateway
        Gateway configured for isolated plugin execution.
    """
    if config is None:
        return open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    return open_gateway(config)


def _run_isolation_worker(
    envelope: IsolationEnvelope,
    result_queue: multiprocessing.Queue[IsolationResult],
) -> None:
    """Execute a plugin in an isolated process and push the result to a queue."""
    try:
        log.info(
            "graph_runtime.plugin.isolation.start name=%s repo=%s commit=%s",
            envelope.plugin_name,
            envelope.repo,
            envelope.commit,
            extra={"graph_run_id": envelope.run_id},
        )
        if envelope.plugin is not None:
            try:
                get_graph_metric_plugin(envelope.plugin.name)
            except KeyError:
                register_graph_metric_plugin(envelope.plugin)
        plugin = plan_graph_metric_plugins((envelope.plugin_name,)).plugins[0]
        gateway = _build_gateway_for_isolation(envelope.gateway_config)
        snapshot = SnapshotRef(repo=envelope.repo, commit=envelope.commit, repo_root=Path())
        runtime = resolve_graph_runtime(
            gateway,
            snapshot,
            GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
        )
        ctx = GraphMetricExecutionContext(
            gateway=gateway,
            runtime=runtime,
            repo=envelope.repo,
            commit=envelope.commit,
            config=None,
            analytics_context=None,
            catalog_provider=None,
            options=envelope.options,
            plugin_name=plugin.name,
            scope=envelope.scope,
            run_id=envelope.run_id,
        )
        plugin_result = _coerce_plugin_result(plugin.run(ctx), plugin.name)
        contracts = run_contracts(
            checkers=plugin.contract_checkers,
            ctx=ctx,
            status="succeeded",
        )
        result_queue.put(
            IsolationResult(
                status="succeeded",
                error=None,
                contracts=contracts,
                row_counts=plugin_result.row_counts if plugin_result is not None else None,
                input_hash=plugin_result.input_hash if plugin_result is not None else None,
                options_hash=plugin_result.options_hash if plugin_result is not None else None,
            )
        )
    except Exception as exc:  # noqa: BLE001 pragma: no cover - defensive
        result_queue.put(
            IsolationResult(
                status="failed",
                error=repr(exc),
                contracts=(),
                row_counts=None,
                input_hash=None,
                options_hash=None,
            )
        )


def _collect_isolation_result(
    process: multiprocessing.Process,
    result_queue: multiprocessing.Queue[IsolationResult],
    timeout_ms: int | None,
    default_input_hash: str | None,
    default_options_hash: str | None,
) -> tuple[
    Literal["succeeded", "failed", "skipped"],
    str | None,
    str | None,
    str | None,
    dict[str, int] | None,
    tuple[PluginContractResult, ...],
]:
    """
    Collect isolation results from a worker process.

    Returns
    -------
    tuple
        (status, error_message, input_hash, options_hash, row_counts, contracts).
    """
    process.join(timeout=(timeout_ms / 1000) if timeout_ms is not None else None)
    status: Literal["succeeded", "failed", "skipped"] = "failed"
    error_message: str | None = None
    input_hash = default_input_hash
    options_hash = default_options_hash
    row_counts: dict[str, int] | None = None
    contracts: tuple[PluginContractResult, ...] = ()
    if process.is_alive():
        process.terminate()
        process.join()
        error_message = "timeout"
        return status, error_message, input_hash, options_hash, row_counts, contracts

    try:
        result = result_queue.get_nowait()
    except Empty:
        result = None
    status = "failed" if result is None else result.status
    if result is None:
        error_message = "no_result"
        return status, error_message, input_hash, options_hash, row_counts, contracts
    if result.error is not None:
        error_message = result.error
        return status, error_message, input_hash, options_hash, row_counts, contracts

    input_hash = result.input_hash or input_hash
    options_hash = result.options_hash or options_hash
    row_counts = result.row_counts
    contracts = result.contracts
    return status, error_message, input_hash, options_hash, row_counts, contracts


def _run_with_timeout(
    func: Callable[[GraphMetricExecutionContext], object | None],
    ctx: GraphMetricExecutionContext,
    timeout_ms: int | None,
) -> object | None:
    """
    Execute a plugin function with an optional timeout.

    Returns
    -------
    object | None
        Result of the callable when it completes within the timeout.

    Raises
    ------
    TimeoutError
        If the callable does not complete within the allotted time.
    """
    if timeout_ms is None:
        return func(ctx)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, ctx)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FuturesTimeout as exc:
            future.cancel()
            message = f"Graph plugin timed out after {timeout_ms} ms"
            raise TimeoutError(message) from exc


def _coerce_plugin_result(result: object | None, plugin_name: str) -> GraphPluginResult | None:
    if result is None:
        return None
    if isinstance(result, GraphPluginResult):
        return result
    log.debug(
        "graph_runtime.plugin.result.unrecognized name=%s result_type=%s",
        plugin_name,
        type(result).__name__,
    )
    return None


def _execute_plugin_isolated(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    run_id: str,
) -> GraphPluginRunRecord:
    """
    Execute a plugin in a separate process.

    Returns
    -------
    GraphPluginRunRecord
        Execution record for the isolated plugin.

    Raises
    ------
    PluginFatalError
        When a fatal contract or execution failure occurs under fail-fast policy.
    """
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    mp_ctx = _select_mp_context()
    envelope = IsolationEnvelope(
        plugin_name=plugin.name,
        plugin=plugin if mp_ctx.start_method() != "fork" else None,
        repo=ctx.repo,
        commit=ctx.commit,
        options=ctx.options,
        scope=ctx.scope,
        gateway_config=getattr(ctx.gateway, "config", None),
        run_id=run_id,
    )
    result_queue: multiprocessing.Queue[IsolationResult] = mp_ctx.queue()
    process = mp_ctx.process(target=_run_isolation_worker, args=(envelope, result_queue))
    process.start()
    (
        status,
        error_message,
        input_hash,
        options_hash,
        row_counts,
        contracts,
    ) = _collect_isolation_result(
        process=process,
        result_queue=result_queue,
        timeout_ms=settings.timeout_ms,
        default_input_hash=settings.input_hash,
        default_options_hash=settings.options_hash,
    )
    record = GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=1,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=datetime.now(tz=UTC),
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        partial=status != "succeeded",
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=input_hash,
        options_hash=options_hash,
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=row_counts,
        contracts=contracts,
        requires_isolation=True,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )
    contract_statuses = {contract.status for contract in contracts}
    if status == "succeeded" and (
        "failed" in contract_statuses or "soft_failed" in contract_statuses
    ):
        record = replace(record, status="failed", error="contract_failed")
        if "failed" in contract_statuses and settings.severity == "fatal" and settings.fail_fast:
            raise PluginFatalError(record, RuntimeError("Contract failure"))
    if record.status == "failed" and settings.severity == "fatal" and settings.fail_fast:
        raise PluginFatalError(record, RuntimeError(error_message or "isolation failure"))
    return record


def _execute_plugin(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    run_id: str,
) -> GraphPluginRunRecord:
    """
    Execute a plugin in-process with retry, timeout, and contract handling.

    Returns
    -------
    GraphPluginRunRecord
        Execution record for the plugin.

    Raises
    ------
    PluginFatalError
        When a fatal failure occurs and fail-fast is enabled.
    """
    if plugin.requires_isolation:
        return _execute_plugin_isolated(plugin=plugin, ctx=ctx, settings=settings, run_id=run_id)
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    attempts = 0
    status: Literal["succeeded", "failed", "skipped"] = "succeeded"
    error_message: str | None = None
    plugin_result: GraphPluginResult | None = None
    while attempts < max(settings.retry_cfg.max_attempts, 1):
        attempts += 1
        try:
            plugin_result = _coerce_plugin_result(
                _run_with_timeout(plugin.run, ctx, settings.timeout_ms),
                plugin.name,
            )
            status = "succeeded"
            error_message = None
            break
        except Exception as exc:
            error_message = repr(exc)
            if settings.severity == "skip_on_error":
                status = "skipped"
                break
            if attempts < max(settings.retry_cfg.max_attempts, 1):
                log.warning(
                    "graph_runtime.plugin.retry name=%s attempt=%d/%d",
                    plugin.name,
                    attempts,
                    max(settings.retry_cfg.max_attempts, 1),
                )
                if settings.retry_cfg.backoff_ms > 0:
                    time.sleep(settings.retry_cfg.backoff_ms / 1000)
                continue
            status = "failed"
            if settings.severity == "fatal" and settings.fail_fast:
                record = GraphPluginRunRecord(
                    name=plugin.name,
                    stage=plugin.stage,
                    severity=settings.severity,
                    status=status,
                    attempts=attempts,
                    timeout_ms=settings.timeout_ms,
                    started_at=started_at,
                    ended_at=datetime.now(tz=UTC),
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    partial=True,
                    run_id=run_id,
                    error=error_message,
                    options=ctx.options,
                    input_hash=settings.input_hash,
                    options_hash=settings.options_hash,
                    version_hash=settings.version_hash,
                    contracts=(),
                    policy_fail_fast=settings.fail_fast,
                )
                raise PluginFatalError(record, exc) from exc
            break
    contracts = run_contracts(
        checkers=settings.contract_checkers,
        ctx=ctx,
        status=status,
    )
    contract_statuses = {result.status for result in contracts}
    input_hash = (
        plugin_result.input_hash
        if plugin_result is not None and plugin_result.input_hash is not None
        else settings.input_hash
    )
    options_hash = (
        plugin_result.options_hash
        if plugin_result is not None and plugin_result.options_hash is not None
        else settings.options_hash
    )
    row_counts = plugin_result.row_counts if plugin_result is not None else None
    if status == "succeeded" and (
        "failed" in contract_statuses or "soft_failed" in contract_statuses
    ):
        status = "failed"
        error_message = "contract_failed"
        if "failed" in contract_statuses and settings.severity == "fatal" and settings.fail_fast:
            record = GraphPluginRunRecord(
                name=plugin.name,
                stage=plugin.stage,
                severity=settings.severity,
                status=status,
                attempts=attempts,
                timeout_ms=settings.timeout_ms,
                started_at=started_at,
                ended_at=datetime.now(tz=UTC),
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                partial=True,
                run_id=run_id,
                error=error_message,
                options=ctx.options,
                input_hash=input_hash,
                options_hash=options_hash,
                version_hash=settings.version_hash,
                skipped_reason=None,
                row_counts=row_counts,
                contracts=contracts,
                requires_isolation=plugin.requires_isolation,
                isolation_kind=plugin.isolation_kind,
                policy_fail_fast=settings.fail_fast,
            )
            raise PluginFatalError(record, RuntimeError("Contract failure"))
    ended_at = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=attempts,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        partial=status != "succeeded",
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=input_hash,
        options_hash=options_hash,
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=row_counts,
        contracts=contracts,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )


def _execute_planned_plugin(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    plan: PluginExecutionPlan,
) -> GraphPluginRunRecord:
    span = plan.telemetry.start_plugin(plugin, plan.run_id, ctx)
    log.info(
        "graph_runtime.plugin.start name=%s repo=%s commit=%s stage=%s",
        plugin.name,
        ctx.repo,
        ctx.commit,
        plugin.stage,
        extra={"graph_run_id": plan.run_id},
    )
    params = RecordParams(
        severity=settings.severity,
        timeout_ms=settings.timeout_ms,
        version_hash=settings.version_hash,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        options=ctx.options,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )
    state = ManifestState(
        plugin_name=plugin.name,
        row_count_tables=plugin.row_count_tables,
        gateway=ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
    )
    if plan.policy.dry_run:
        record = dry_run_record(plugin=plugin, params=params, run_id=plan.run_id)
    elif plan.policy.skip_on_unchanged and is_unchanged(plan.prior_manifest, state):
        record = skip_record(plugin=plugin, params=params, reason="unchanged", run_id=plan.run_id)
    else:
        record = _execute_plugin(plugin=plugin, ctx=ctx, settings=settings, run_id=plan.run_id)
    plan.telemetry.finish_plugin(span, record)
    plan.telemetry.record_metrics(record, plan.scope)
    log.info(
        "graph_runtime.plugin.finish name=%s stage=%s status=%s duration_ms=%.2f attempts=%d",
        record.name,
        record.stage,
        record.status,
        record.duration_ms,
        record.attempts,
        extra={
            "metric": "graph_runtime",
            "op": record.name,
            "duration_ms": record.duration_ms,
            "use_gpu": ctx.runtime.use_gpu,
            "features": ctx.runtime.options.features,
            "plugin_status": record.status,
            "plugin_started_at": record.started_at.isoformat(),
            "plugin_ended_at": record.ended_at.isoformat(),
            "plugin_stage": record.stage,
            "plugin_attempts": record.attempts,
            "plugin_timeout_ms": record.timeout_ms,
            "plugin_severity": record.severity,
            "plugin_contracts": [c.status for c in record.contracts],
            "graph_run_id": plan.run_id,
        },
    )
    return record


def run_graph_plugin_batch(
    *,
    plan: PluginExecutionPlan,
    context: BatchContext,
) -> list[GraphPluginRunRecord]:
    """
    Execute all plugins in a PluginExecutionPlan and return run records.

    Returns
    -------
    list[GraphPluginRunRecord]
        Records emitted for each plugin in the plan.

    Raises
    ------
    PluginFatalError
        When a fatal plugin failure occurs and fail-fast is enabled.
    """
    records: list[GraphPluginRunRecord] = []
    scratch = GraphRuntimeScratch()

    def _run_single_plugin(plugin: GraphMetricPlugin) -> GraphPluginRunRecord:
        settings = plan.settings_by_plugin[plugin.name]
        options = plan.options_by_plugin.get(plugin.name)
        plugin_ctx = GraphMetricExecutionContext(
            gateway=context.gateway,
            runtime=context.runtime,
            repo=plan.repo,
            commit=plan.commit,
            config=context.cfg,
            analytics_context=context.analytics_context,
            catalog_provider=context.catalog_provider,
            options=options,
            plugin_name=plugin.name,
            scope=plan.scope,
            run_id=plan.run_id,
            scratch=scratch,
        )
        return _execute_planned_plugin(plugin=plugin, ctx=plugin_ctx, settings=settings, plan=plan)

    try:
        for plugin in plan.plugins:
            try:
                record = _run_single_plugin(plugin)
            except PluginFatalError as exc:
                records.append(exc.record)
                raise
            records.append(record)
    finally:
        scratch.cleanup()
    return records


__all__ = [
    "IsolationEnvelope",
    "IsolationResult",
    "MPContext",
    "PluginFatalError",
    "run_graph_plugin_batch",
]
