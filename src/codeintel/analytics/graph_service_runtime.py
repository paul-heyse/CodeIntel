"""Graph runtime context utilities shared across analytics graph metrics."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graphs.contracts import (
    ContractChecker,
    PluginContractResult,
    run_contract_checkers,
)
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphMetricPluginSkip,
    get_graph_metric_plugin,
    plan_graph_metric_plugins,
    register_graph_metric_plugin,
    resolve_plugin_options,
)
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphPluginRetryPolicy, GraphRunScope
from codeintel.storage.gateway import StorageConfig, open_gateway, open_memory_gateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

DEFAULT_BETWEENNESS_SAMPLE = 500


def _hash_json(payload: object) -> str:
    """
    Stable JSON hash helper for telemetry labels.

    Returns
    -------
    str
        Hex digest of the serialized payload.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compute_options_hash(options: object | None) -> str:
    """
    Compute a stable hash for plugin options payloads.

    Returns
    -------
    str
        Hex digest of normalized options.
    """
    return _hash_json(options or {})


@dataclass(frozen=True)
class GraphPluginRunRecord:
    """Per-plugin execution telemetry."""

    name: str
    stage: str
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    status: Literal["succeeded", "failed", "skipped"]
    attempts: int
    timeout_ms: int | None
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    partial: bool
    run_id: str
    error: str | None = None
    options: object | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    version_hash: str | None = None
    skipped_reason: str | None = None
    row_counts: dict[str, int] | None = None
    contracts: tuple[PluginContractResult, ...] = ()
    requires_isolation: bool = False
    isolation_kind: str | None = None
    policy_fail_fast: bool = False


@dataclass(frozen=True)
class GraphPluginRunReport:
    """Aggregate execution report for a plugin batch."""

    repo: str
    commit: str
    records: tuple[GraphPluginRunRecord, ...]
    scope: GraphRunScope
    run_id: str
    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]


class GraphRuntimeTelemetry(Protocol):
    """Optional observability hooks for graph runtime events."""

    def start_plugin(
        self, plugin: GraphMetricPlugin, run_id: str, ctx: GraphMetricExecutionContext
    ) -> object: ...

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None: ...

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None: ...


class _NoOpGraphRuntimeTelemetry:
    """No-op telemetry implementation."""

    @staticmethod
    def start_plugin(
        _plugin: GraphMetricPlugin, _run_id: str, _ctx: GraphMetricExecutionContext
    ) -> None:
        return None

    @staticmethod
    def finish_plugin(span: object, record: GraphPluginRunRecord) -> None:
        _ = span
        _ = record

    @staticmethod
    def record_metrics(_record: GraphPluginRunRecord, _scope: GraphRunScope) -> None:
        return None


class _OtelGraphRuntimeTelemetry:
    """OpenTelemetry-backed telemetry for graph plugins."""

    def __init__(self) -> None:
        self.tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        self.duration_ms = meter.create_histogram(
            "graph_plugin_duration_ms",
            unit="ms",
            description="Duration of graph metric plugins",
        )
        self.status_counter = meter.create_counter(
            "graph_plugin_status_total",
            description="Count of graph metric plugin executions by status",
        )
        self.retry_counter = meter.create_counter(
            "graph_plugin_retries_total",
            description="Count of retry attempts for graph metric plugins",
        )
        self.skip_counter = meter.create_counter(
            "graph_plugin_skipped_total",
            description="Count of skipped graph metric plugins by reason",
        )

    def start_plugin(
        self, plugin: GraphMetricPlugin, run_id: str, ctx: GraphMetricExecutionContext
    ) -> object:
        scope_payload = {
            "paths": ctx.scope.paths,
            "modules": ctx.scope.modules,
            "time_window": (
                (
                    ctx.scope.time_window[0].isoformat(),
                    ctx.scope.time_window[1].isoformat(),
                )
                if ctx.scope.time_window is not None
                else None
            ),
        }
        attributes = {
            "graph.plugin": plugin.name,
            "graph.stage": plugin.stage,
            "graph.run_id": run_id,
            "graph.scope.paths": len(ctx.scope.paths),
            "graph.scope.modules": len(ctx.scope.modules),
            "graph.scope.time_window": ctx.scope.time_window is not None,
            "graph.repo": ctx.repo,
            "graph.commit": ctx.commit,
            "graph.scope_hash": _hash_json(scope_payload) if scope_payload else "none",
            "graph.options_hash": _hash_json(ctx.options) if ctx.options is not None else "none",
        }
        return self.tracer.start_span("graph.plugin", attributes=attributes)

    @staticmethod
    def finish_plugin(span: object, record: GraphPluginRunRecord) -> None:
        span_obj = cast("trace.Span", span)
        span_obj.set_attribute("graph.status", record.status)
        span_obj.set_attribute("graph.attempts", record.attempts)
        span_obj.set_attribute("graph.severity", record.severity)
        span_obj.set_attribute("graph.partial", record.partial)
        span_obj.set_attribute("graph.requires_isolation", record.requires_isolation)
        span_obj.set_attribute("graph.policy_fail_fast", record.policy_fail_fast)
        span_obj.set_attribute("graph.input_hash", record.input_hash or "none")
        span_obj.set_attribute("graph.version_hash", record.version_hash or "none")
        span_obj.set_attribute("graph.timeout_ms", record.timeout_ms or 0)
        span_obj.set_attribute(
            "graph.options_hash",
            _hash_json(record.options) if record.options is not None else "none",
        )
        if record.isolation_kind is not None:
            span_obj.set_attribute("graph.isolation_kind", record.isolation_kind)
        if record.error is not None:
            span_obj.record_exception(Exception(record.error))
            span_obj.set_status(Status(StatusCode.ERROR, record.error))
        else:
            span_obj.set_status(Status(StatusCode.OK))
        span_obj.end()

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        scope_present = bool(scope.paths or scope.modules or scope.time_window)
        scope_payload = {
            "paths": scope.paths,
            "modules": scope.modules,
            "time_window": scope.time_window,
        }
        scope_hash = _hash_json(scope_payload) if scope_present else "none"
        options_hash = _hash_json(record.options) if record.options is not None else "none"
        attributes = {
            "plugin": record.name,
            "stage": record.stage,
            "severity": record.severity,
            "status": record.status,
            "requires_isolation": record.requires_isolation,
            "isolation_kind": record.isolation_kind or "none",
            "scope_paths": len(scope.paths),
            "scope_modules": len(scope.modules),
            "scope_time_window": scope.time_window is not None,
            "scope_present": scope_present,
            "scope_hash": scope_hash,
            "options_hash": options_hash,
            "policy_fail_fast": record.policy_fail_fast,
        }
        self.duration_ms.record(record.duration_ms, attributes=attributes)
        self.status_counter.add(1, attributes=attributes)
        if record.attempts > 1:
            self.retry_counter.add(record.attempts - 1, attributes=attributes)
        if record.status == "skipped":
            self.skip_counter.add(
                1, attributes={**attributes, "skip_reason": record.skipped_reason or "unspecified"}
            )


@dataclass(frozen=True)
class _IsolationEnvelope:
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
class _IsolationResult:
    """Serialized outputs from isolated plugin execution."""

    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    contracts: tuple[PluginContractResult, ...]
    row_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class _MPContext:
    """Minimal multiprocessing context for isolation support."""

    process_factory: Callable[..., multiprocessing.Process]
    queue_factory: Callable[[], multiprocessing.Queue[_IsolationResult]]
    start_method_name: str

    def process(
        self, target: Callable[..., object], args: tuple[object, ...]
    ) -> multiprocessing.Process:
        return self.process_factory(target=target, args=args)

    def queue(self) -> multiprocessing.Queue[_IsolationResult]:
        return self.queue_factory()

    def start_method(self) -> str:
        return self.start_method_name


class _PluginFatalError(Exception):
    """Fatal plugin failure while respecting fail-fast semantics."""

    def __init__(self, record: GraphPluginRunRecord, original: Exception) -> None:
        super().__init__(str(original))
        self.record = record


@dataclass(frozen=True)
class _PluginExecutionSettings:
    """Resolved execution policy for a plugin."""

    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None
    contract_checkers: tuple[ContractChecker, ...]


@dataclass(frozen=True)
class _PluginExecutionPlan:
    """Execution context shared across plugin runs for telemetry and policy."""

    policy: GraphPluginPolicy
    prior_manifest: dict[str, dict[str, object]] | None
    telemetry: GraphRuntimeTelemetry
    run_id: str
    scope: GraphRunScope


@dataclass(frozen=True)
class GraphPluginRunOptions:
    """Optional controls for plugin execution."""

    plugin_options: dict[str, dict[str, object]] | None = None
    manifest_path: Path | None = None
    scope: GraphRunScope | None = None
    dry_run: bool | None = None


def _resolve_plugin_options_map(
    plugins: Sequence[GraphMetricPlugin],
    cfg_options: dict[str, dict[str, object]],
    runtime_options: dict[str, dict[str, object]],
) -> dict[str, object | None]:
    """
    Merge and validate plugin options from config and runtime.

    Raises
    ------
    ValueError
        When options are supplied for plugins not present in the plan.

    Returns
    -------
    dict[str, object | None]
        Validated and merged options keyed by plugin name.
    """
    allowed_plugins = {plugin.name for plugin in plugins}
    unknown_option_plugins = (
        set(cfg_options.keys()) | set(runtime_options.keys())
    ) - allowed_plugins
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


@dataclass(frozen=True)
class GraphContext:
    """Execution context for graph computations."""

    repo: str
    commit: str
    now: datetime | None = None
    betweenness_sample: int = 500
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    use_gpu: bool = False
    community_detection_limit: int | None = None

    def resolved_now(self) -> datetime:
        """
        Return a concrete timestamp, defaulting to UTC now when unset.

        Returns
        -------
        datetime
            Existing timestamp or the current UTC time.
        """
        return self.now or datetime.now(tz=UTC)


@dataclass(frozen=True)
class GraphContextSpec:
    """Specification for normalizing graph contexts."""

    repo: str
    commit: str
    use_gpu: bool
    metrics_cfg: GraphMetricsStepConfig | None = None
    ctx: GraphContext | None = None
    now: datetime | None = None
    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    pagerank_weight: str | None = None
    betweenness_weight: str | None = None
    seed: int | None = None
    community_detection_limit: int | None = None


@dataclass(frozen=True)
class GraphContextCaps:
    """Optional caps for graph context derivation."""

    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    community_detection_limit: int | None = None


@dataclass
class GraphServiceRuntime:
    """Lightweight orchestrator for graph analytics using a shared runtime."""

    gateway: StorageGateway
    runtime: GraphRuntime
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    telemetry: GraphRuntimeTelemetry | None = None

    def run_plugins(
        self,
        plugin_names: Sequence[str],
        *,
        cfg: GraphMetricsStepConfig | None = None,
        target: tuple[str, str] | None = None,
        run_options: GraphPluginRunOptions | None = None,
    ) -> GraphPluginRunReport:
        """
        Execute a sequence of graph metric plugins against this runtime.

        Parameters
        ----------
        plugin_names:
            Names of plugins to execute, in order.
        cfg:
            Optional graph metrics configuration; provided to plugins via
            GraphMetricExecutionContext.
        target:
            Optional (repo, commit) override when config is not supplied.
        run_options:
            Optional execution controls (per-plugin options, manifest path, scopes).

        Raises
        ------
        ValueError
            If neither a config nor runtime snapshot is available to derive repo/commit.
        _PluginFatalError
            When a fatal plugin failure occurs and fail-fast is enabled.

        Returns
        -------
        GraphPluginRunReport
            Telemetry for executed plugins.
        """
        policy = cfg.plugin_policy if cfg is not None else GraphPluginPolicy()
        if run_options is not None and run_options.dry_run is not None:
            policy = replace(policy, dry_run=run_options.dry_run)
        run_id = uuid.uuid4().hex
        manifest_path = run_options.manifest_path if run_options is not None else None
        prior_manifest = _load_prior_manifest(manifest_path)
        if cfg is None and target is None and self.runtime.options.snapshot is None:
            message = "Graph runtime missing snapshot; cannot derive repo/commit"
            raise ValueError(message)
        resolved_target = _resolve_target(cfg, target, self.runtime)
        resolved_scope = (
            run_options.scope
            if run_options is not None and run_options.scope is not None
            else cfg.scope
            if cfg is not None
            else GraphRunScope()
        )
        execution_plan = _PluginExecutionPlan(
            policy=policy,
            prior_manifest=prior_manifest,
            telemetry=self.telemetry or _OtelGraphRuntimeTelemetry(),
            run_id=run_id,
            scope=resolved_scope,
        )
        plan = plan_graph_metric_plugins(plugin_names)
        plugins: tuple[GraphMetricPlugin, ...] = plan.plugins
        resolved_options = _resolve_plugin_options_map(
            plugins=plugins,
            cfg_options=cfg.plugin_options if cfg is not None else {},
            runtime_options=(run_options.plugin_options if run_options is not None else {}) or {},
        )
        records: list[GraphPluginRunRecord] = []

        def _run_single_plugin(plugin: GraphMetricPlugin) -> GraphPluginRunRecord:
            options = resolved_options.get(plugin.name)
            settings = _PluginExecutionSettings(
                severity=_effective_severity(plugin, policy),
                retry_cfg=policy.retries.get(plugin.name, GraphPluginRetryPolicy()),
                timeout_ms=_effective_timeout(plugin, policy),
                fail_fast=policy.fail_fast,
                input_hash=_compute_input_hash(
                    resolved_target[0],
                    resolved_target[1],
                    plugin.version_hash,
                    plugin.name,
                    options,
                ),
                options_hash=_compute_options_hash(options),
                version_hash=plugin.version_hash,
                contract_checkers=plugin.contract_checkers,
            )
            plugin_ctx = GraphMetricExecutionContext(
                gateway=self.gateway,
                runtime=self.runtime,
                repo=resolved_target[0],
                commit=resolved_target[1],
                config=cfg,
                analytics_context=self.analytics_context,
                catalog_provider=self.catalog_provider,
                options=options,
                plugin_name=plugin.name,
                scope=resolved_scope,
                run_id=execution_plan.run_id,
            )
            return _execute_planned_plugin(
                plugin=plugin,
                ctx=plugin_ctx,
                settings=settings,
                plan=execution_plan,
            )

        for plugin in plugins:
            try:
                record = _run_single_plugin(plugin)
            except _PluginFatalError as exc:
                records.append(exc.record)
                raise
            records.append(record)
        report = GraphPluginRunReport(
            repo=resolved_target[0],
            commit=resolved_target[1],
            records=tuple(records),
            scope=resolved_scope,
            run_id=run_id,
            plan_id=plan.plan_id,
            ordered_plugins=plan.ordered_names,
            skipped_plugins=plan.skipped_plugins,
            dep_graph=plan.dep_graph,
        )
        if manifest_path is not None:
            _write_plugin_manifest(manifest_path, report)
        return report

    def compute_graph_metrics(
        self, cfg: GraphMetricsStepConfig, *, filters: object | None = None
    ) -> None:
        """Compute core function/module graph metrics."""
        if filters is not None:
            log.debug("Graph metric filters are ignored in plugin-driven execution.")
        self.run_plugins(("core_graph_metrics",), cfg=cfg)

    def compute_graph_metrics_ext(self, *, repo: str, commit: str) -> None:
        """Compute extended function and module graph metrics."""
        self.run_plugins(
            ("graph_metrics_functions_ext", "graph_metrics_modules_ext"),
            target=(repo, commit),
        )

    def compute_symbol_metrics(self, *, repo: str, commit: str) -> None:
        """Compute symbol graph metrics for modules and functions."""
        self.run_plugins(
            ("symbol_graph_metrics_modules", "symbol_graph_metrics_functions"),
            target=(repo, commit),
        )

    def compute_subsystem_metrics(self, *, repo: str, commit: str) -> None:
        """Compute subsystem-level graph metrics."""
        self.run_plugins(("subsystem_graph_metrics",), target=(repo, commit))

    def compute_graph_stats(self, *, repo: str, commit: str) -> None:
        """Compute global graph statistics."""
        self.run_plugins(("graph_stats",), target=(repo, commit))


def build_graph_context(
    cfg: GraphMetricsStepConfig,
    *,
    now: datetime | None = None,
    caps: GraphContextCaps | None = None,
    use_gpu: bool = False,
) -> GraphContext:
    """
    Construct a GraphContext from GraphMetricsStepConfig with optional caps.

    Parameters
    ----------
    cfg :
        Graph metrics configuration values.
    now :
        Optional timestamp; defaults to UTC now when omitted.
    caps :
        Optional container for sampling caps and community detection limit.
    use_gpu :
        Whether to prefer GPU-backed NetworkX execution when available.

    Returns
    -------
    GraphContext
        Graph context with caps and seeds applied.
    """
    resolved_caps = caps or GraphContextCaps()
    betweenness_sample = cfg.max_betweenness_sample or DEFAULT_BETWEENNESS_SAMPLE
    if resolved_caps.betweenness_cap is not None:
        betweenness_sample = min(betweenness_sample, resolved_caps.betweenness_cap)
    eigen_max_iter = (
        cfg.eigen_max_iter
        if resolved_caps.eigen_cap is None
        else min(cfg.eigen_max_iter, resolved_caps.eigen_cap)
    )
    return GraphContext(
        repo=cfg.repo,
        commit=cfg.commit,
        now=now,
        betweenness_sample=betweenness_sample,
        eigen_max_iter=eigen_max_iter,
        seed=cfg.seed,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_weight=cfg.betweenness_weight,
        use_gpu=use_gpu,
        community_detection_limit=resolved_caps.community_detection_limit,
    )


def resolve_graph_context(
    spec: GraphContextSpec,
) -> GraphContext:
    """
    Normalize a GraphContext to the target repo/commit and backend preferences.

    Parameters
    ----------
    spec :
        Context specification describing the repo/commit, backend preference, and
        optional overrides.

    Returns
    -------
    GraphContext
        Context aligned to the provided repo, commit, and backend preferences.
    """
    base_now = spec.now or datetime.now(tz=UTC)
    resolved = _base_context(spec, base_now)
    return _normalize_context(spec, resolved, base_now)


def _base_context(spec: GraphContextSpec, base_now: datetime) -> GraphContext:
    if spec.ctx is not None:
        return spec.ctx
    if spec.metrics_cfg is not None:
        caps = GraphContextCaps(
            betweenness_cap=spec.betweenness_cap,
            eigen_cap=spec.eigen_cap,
            community_detection_limit=spec.community_detection_limit,
        )
        return build_graph_context(
            spec.metrics_cfg,
            now=base_now,
            caps=caps,
            use_gpu=spec.use_gpu,
        )
    return GraphContext(
        repo=spec.repo,
        commit=spec.commit,
        now=base_now,
        betweenness_sample=spec.betweenness_cap or DEFAULT_BETWEENNESS_SAMPLE,
        eigen_max_iter=spec.eigen_cap or DEFAULT_BETWEENNESS_SAMPLE,
        seed=spec.seed or 0,
        pagerank_weight=spec.pagerank_weight or "weight",
        betweenness_weight=spec.betweenness_weight or "weight",
        use_gpu=spec.use_gpu,
        community_detection_limit=spec.community_detection_limit,
    )


def _normalize_context(
    spec: GraphContextSpec,
    ctx: GraphContext,
    base_now: datetime,
) -> GraphContext:
    normalized = ctx
    if ctx.repo != spec.repo or ctx.commit != spec.commit:
        normalized = replace(normalized, repo=spec.repo, commit=spec.commit)
    if normalized.use_gpu != spec.use_gpu:
        normalized = replace(normalized, use_gpu=spec.use_gpu)
    if spec.betweenness_cap is not None and normalized.betweenness_sample > spec.betweenness_cap:
        normalized = replace(normalized, betweenness_sample=spec.betweenness_cap)
    if spec.eigen_cap is not None and normalized.eigen_max_iter > spec.eigen_cap:
        normalized = replace(normalized, eigen_max_iter=spec.eigen_cap)
    if spec.pagerank_weight is not None and normalized.pagerank_weight != spec.pagerank_weight:
        normalized = replace(normalized, pagerank_weight=spec.pagerank_weight)
    if (
        spec.betweenness_weight is not None
        and normalized.betweenness_weight != spec.betweenness_weight
    ):
        normalized = replace(normalized, betweenness_weight=spec.betweenness_weight)
    if spec.seed is not None and normalized.seed != spec.seed:
        normalized = replace(normalized, seed=spec.seed)
    if normalized.now is None:
        normalized = replace(normalized, now=base_now)
    if (
        spec.community_detection_limit is not None
        and normalized.community_detection_limit != spec.community_detection_limit
    ):
        normalized = replace(normalized, community_detection_limit=spec.community_detection_limit)
    return normalized


def _effective_severity(
    plugin: GraphMetricPlugin, policy: GraphPluginPolicy
) -> Literal["fatal", "soft_fail", "skip_on_error"]:
    override = policy.severity_overrides.get(plugin.name)
    if override is not None:
        return override
    severity = getattr(plugin, "severity", None)
    return severity if severity is not None else policy.default_severity


def _effective_timeout(plugin: GraphMetricPlugin, policy: GraphPluginPolicy) -> int | None:
    override = policy.timeouts_ms.get(plugin.name)
    if override is not None:
        return override
    hints = getattr(plugin, "resource_hints", None)
    return hints.max_runtime_ms if hints is not None else None


def _resolve_target(
    cfg: GraphMetricsStepConfig | None,
    target: tuple[str, str] | None,
    runtime: GraphRuntime,
) -> tuple[str, str]:
    if cfg is not None:
        return cfg.repo, cfg.commit
    if target is not None:
        return target
    snapshot = runtime.options.snapshot
    if snapshot is None:
        message = "Graph runtime missing snapshot; cannot derive repo/commit"
        raise ValueError(message)
    return snapshot.repo, snapshot.commit


def _execute_planned_plugin(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: _PluginExecutionSettings,
    plan: _PluginExecutionPlan,
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
    if plan.policy.dry_run:
        record = _dry_run_record(plugin, settings, ctx.options, plan.run_id)
    elif plan.policy.skip_on_unchanged and _is_unchanged(
        plan.prior_manifest or {},
        plugin.name,
        settings.input_hash,
        settings.options_hash,
    ):
        record = _skip_record(plugin, settings, ctx.options, reason="unchanged", run_id=plan.run_id)
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


def _run_with_timeout(
    func: Callable[[GraphMetricExecutionContext], None],
    ctx: GraphMetricExecutionContext,
    timeout_ms: int | None,
) -> None:
    if timeout_ms is None:
        func(ctx)
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, ctx)
        try:
            future.result(timeout=timeout_ms / 1000)
        except FuturesTimeout as exc:
            future.cancel()
            message = f"Graph plugin timed out after {timeout_ms} ms"
            raise TimeoutError(message) from exc


def _build_gateway_for_isolation(config: StorageConfig | None) -> StorageGateway:
    if config is None:
        return open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    return open_gateway(config)


def _select_mp_context() -> _MPContext:
    """
    Prefer forked processes to preserve in-memory plugin registration.

    Falls back to the default start method when fork is unavailable (e.g., Windows).

    Returns
    -------
    _MPContext
        Multiprocessing context using fork when available.
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
        "Callable[[], multiprocessing.Queue[_IsolationResult]]",
        base_ctx_any.Queue,
    )
    return _MPContext(
        process_factory=process_factory,
        queue_factory=queue_factory,
        start_method_name=base_ctx.get_start_method(),
    )


def _run_isolation_worker(
    envelope: _IsolationEnvelope, result_queue: multiprocessing.Queue[_IsolationResult]
) -> None:
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
        plugin.run(ctx)
        contracts = run_contract_checkers(ctx=ctx, checkers=plugin.contract_checkers)
        result_queue.put(
            _IsolationResult(
                status="succeeded",
                error=None,
                contracts=contracts,
                row_counts=None,
            )
        )
    except Exception as exc:  # noqa: BLE001 pragma: no cover - defensive
        result_queue.put(
            _IsolationResult(
                status="failed",
                error=repr(exc),
                contracts=(),
                row_counts=None,
            )
        )


def _execute_plugin_isolated(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: _PluginExecutionSettings,
    run_id: str,
) -> GraphPluginRunRecord:
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    mp_ctx = _select_mp_context()
    envelope = _IsolationEnvelope(
        plugin_name=plugin.name,
        plugin=plugin if mp_ctx.start_method() != "fork" else None,
        repo=ctx.repo,
        commit=ctx.commit,
        options=ctx.options,
        scope=ctx.scope,
        gateway_config=getattr(ctx.gateway, "config", None),
        run_id=run_id,
    )
    result_queue: multiprocessing.Queue[_IsolationResult] = mp_ctx.queue()
    process = mp_ctx.process(target=_run_isolation_worker, args=(envelope, result_queue))
    process.start()
    timeout_sec = settings.timeout_ms / 1000 if settings.timeout_ms is not None else None
    process.join(timeout=timeout_sec)
    result: _IsolationResult | None = None
    status: Literal["succeeded", "failed", "skipped"] = "failed"
    error_message: str | None = None
    if process.is_alive():
        process.terminate()
        process.join()
        error_message = "timeout"
    else:
        try:
            result = result_queue.get_nowait()
        except Empty:
            result = None
        status = "failed" if result is None else result.status
        if result is None:
            error_message = "no_result"
        elif result.error is not None:
            error_message = result.error
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    partial = status != "succeeded"
    record = GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=1,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=datetime.now(tz=UTC),
        duration_ms=duration_ms,
        partial=partial,
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=result.row_counts if result is not None else None,
        contracts=result.contracts if result is not None else (),
        requires_isolation=True,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )
    if status == "failed" and settings.severity == "fatal" and settings.fail_fast:
        raise _PluginFatalError(record, RuntimeError(error_message or "isolation failure"))
    return record


def _execute_plugin(
    *,
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: _PluginExecutionSettings,
    run_id: str,
) -> GraphPluginRunRecord:
    if plugin.requires_isolation:
        return _execute_plugin_isolated(plugin=plugin, ctx=ctx, settings=settings, run_id=run_id)
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    attempts = 0
    status: Literal["succeeded", "failed", "skipped"] = "succeeded"
    error_message: str | None = None
    max_attempts = max(settings.retry_cfg.max_attempts, 1)
    while attempts < max_attempts:
        attempts += 1
        try:
            _run_with_timeout(plugin.run, ctx, settings.timeout_ms)
            status = "succeeded"
            error_message = None
            break
        except Exception as exc:
            error_message = repr(exc)
            if settings.severity == "skip_on_error":
                status = "skipped"
                break
            if attempts < max_attempts:
                log.warning(
                    "graph_runtime.plugin.retry name=%s attempt=%d/%d",
                    plugin.name,
                    attempts,
                    max_attempts,
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
                raise _PluginFatalError(record, exc) from exc
            break
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    ended_at = datetime.now(tz=UTC)
    partial = status != "succeeded"
    contracts = _run_contracts(settings.contract_checkers, ctx, status)
    contract_failure = any(result.status == "failed" for result in contracts)
    contract_soft_failure = any(result.status == "soft_failed" for result in contracts)
    if status == "succeeded" and (contract_failure or contract_soft_failure):
        status = "failed"
        partial = True
        error_message = "contract_failed"
        if contract_failure and settings.severity == "fatal" and settings.fail_fast:
            record = GraphPluginRunRecord(
                name=plugin.name,
                stage=plugin.stage,
                severity=settings.severity,
                status=status,
                attempts=attempts,
                timeout_ms=settings.timeout_ms,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                partial=partial,
                run_id=run_id,
                error=error_message,
                options=ctx.options,
                input_hash=settings.input_hash,
                options_hash=settings.options_hash,
                version_hash=settings.version_hash,
                skipped_reason=None,
                row_counts=None,
                contracts=contracts,
                requires_isolation=plugin.requires_isolation,
                isolation_kind=plugin.isolation_kind,
                policy_fail_fast=settings.fail_fast,
            )
            raise _PluginFatalError(record, RuntimeError("Contract failure"))
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=attempts,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        partial=partial,
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=None,
        contracts=contracts,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )


def _write_plugin_manifest(path: Path, report: GraphPluginRunReport) -> None:
    payload = _report_to_payload(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _report_to_payload(report: GraphPluginRunReport) -> dict[str, object]:
    return {
        "repo": report.repo,
        "commit": report.commit,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_id": report.run_id,
        "plan": {
            "plan_id": report.plan_id,
            "ordered_plugins": list(report.ordered_plugins),
            "skipped_plugins": [
                {"name": skipped.name, "reason": skipped.reason}
                for skipped in report.skipped_plugins
            ],
            "dep_graph": {name: list(deps) for name, deps in report.dep_graph.items()},
        },
        "scope": {
            "paths": list(report.scope.paths),
            "modules": list(report.scope.modules),
            "time_window": (
                (
                    report.scope.time_window[0].isoformat(),
                    report.scope.time_window[1].isoformat(),
                )
                if report.scope.time_window is not None
                else None
            ),
        },
        "records": [
            {
                "name": record.name,
                "stage": record.stage,
                "severity": record.severity,
                "status": record.status,
                "attempts": record.attempts,
                "timeout_ms": record.timeout_ms,
                "started_at": record.started_at.isoformat(),
                "ended_at": record.ended_at.isoformat(),
                "duration_ms": record.duration_ms,
                "partial": record.partial,
                "run_id": record.run_id,
                "error": record.error,
                "input_hash": record.input_hash,
                "options_hash": record.options_hash,
                "version_hash": record.version_hash,
                "skipped_reason": record.skipped_reason,
                "row_counts": record.row_counts,
                "requires_isolation": record.requires_isolation,
                "isolation_kind": record.isolation_kind,
                "contracts": [
                    {
                        "name": contract.name,
                        "status": contract.status,
                        "message": contract.message,
                    }
                    for contract in record.contracts
                ],
                "policy_fail_fast": record.policy_fail_fast,
            }
            for record in report.records
        ],
    }


def _load_prior_manifest(manifest_path: Path | None) -> dict[str, dict[str, object]] | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        return {
            record.get("name", "unknown"): record for record in records if isinstance(record, dict)
        }
    except (OSError, json.JSONDecodeError):
        return None


def _compute_input_hash(
    repo: str, commit: str, version_hash: str | None, plugin_name: str, options: object | None
) -> str:
    parts = {
        "repo": repo,
        "commit": commit,
        "plugin": plugin_name,
        "version_hash": version_hash or "0",
        "options": options,
    }
    serialized = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_unchanged(
    prior_manifest: dict[str, dict[str, object]],
    plugin_name: str,
    input_hash: str | None,
    options_hash: str | None,
) -> bool:
    prior = prior_manifest.get(plugin_name)
    if not (
        input_hash is not None
        and prior is not None
        and prior.get("status") == "succeeded"
        and prior.get("input_hash") == input_hash
        and (options_hash is None or prior.get("options_hash") == options_hash)
    ):
        return False
    prior_rows = prior.get("row_counts")
    if prior_rows is None:
        return True
    return isinstance(prior_rows, dict)


def _dry_run_record(
    plugin: GraphMetricPlugin,
    settings: _PluginExecutionSettings,
    options: object | None,
    run_id: str,
) -> GraphPluginRunRecord:
    now_ts = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status="skipped",
        attempts=0,
        timeout_ms=settings.timeout_ms,
        started_at=now_ts,
        ended_at=now_ts,
        duration_ms=0.0,
        partial=False,
        run_id=run_id,
        error=None,
        options=options,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        version_hash=settings.version_hash,
        skipped_reason="dry_run",
        row_counts=None,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )


def _skip_record(
    plugin: GraphMetricPlugin,
    settings: _PluginExecutionSettings,
    options: object | None,
    reason: str,
    run_id: str,
) -> GraphPluginRunRecord:
    now_ts = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status="skipped",
        attempts=0,
        timeout_ms=settings.timeout_ms,
        started_at=now_ts,
        ended_at=now_ts,
        duration_ms=0.0,
        partial=False,
        run_id=run_id,
        error=None,
        options=options,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        version_hash=settings.version_hash,
        skipped_reason=reason,
        row_counts=None,
        contracts=(),
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )


def _run_contracts(
    checkers: tuple[ContractChecker, ...],
    ctx: GraphMetricExecutionContext,
    status: Literal["succeeded", "failed", "skipped"],
) -> tuple[PluginContractResult, ...]:
    if not checkers or status != "succeeded":
        return ()
    return run_contract_checkers(ctx=ctx, checkers=checkers)


__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphServiceRuntime",
    "build_graph_context",
    "resolve_graph_context",
]
