"""Generic analytics plugin planning and execution harness."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.contracts import PluginContractResult, run_contract_checkers
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    GraphRuntimeScratch,
    get_graph_metric_plugin,
)
from codeintel.analytics.graphs.runtime.analytics_adapter import _meta_from_graph_record
from codeintel.analytics.graphs.runtime.execution import (
    PluginFatalError,
)
from codeintel.analytics.graphs.runtime.execution import (
    _execute_plugin as _run_graph_execution_plugin,
)
from codeintel.analytics.graphs.runtime.manifest import (
    ManifestState,
    RecordParams,
    dry_run_record,
    hash_json,
    is_unchanged,
    skip_record,
)
from codeintel.analytics.graphs.runtime.model import GraphPluginRunRecord
from codeintel.analytics.graphs.runtime.planning import PluginExecutionSettings, PluginSeverity
from codeintel.analytics.graphs.runtime.telemetry import (
    GraphRuntimeTelemetry,
    NoOpGraphRuntimeTelemetry,
)
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    plan_analytics_plugins,
)
from codeintel.analytics.runtime_manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
)
from codeintel.config import (
    BehavioralCoverageStepConfig,
    ConfigDataFlowStepConfig,
    CoverageAnalyticsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    ExternalDependenciesStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ProfilesAnalyticsStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:  # pragma: no cover
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
else:  # pragma: no cover
    class FunctionCatalogProvider:  # type: ignore[too-many-instance-attributes]
        ...

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsPluginRunOptions:
    """Optional per-run controls for analytics plugins."""

    plugin_options: dict[str, dict[str, object]] | None = None
    manifest_path: Path | None = None
    scope: GraphRunScope | None = None
    dry_run: bool | None = None


@dataclass(frozen=True)
class AnalyticsPluginExecutionSettings:
    """Resolved execution policy and hashes for a single plugin."""

    name: str
    severity: str
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None
    options_hash: str | None
    version_hash: str | None


@dataclass(frozen=True)
class AnalyticsPluginExecutionPlan:
    """Execution plan for a batch of analytics plugins."""

    plan_id: str
    run_id: str
    repo: str
    commit: str
    policy: GraphPluginPolicy
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    scope: GraphRunScope
    plugins: tuple[AnalyticsPlugin, ...]
    ordered_names: tuple[str, ...]
    skipped: tuple[AnalyticsSkippedStep, ...]
    dep_graph: dict[str, tuple[str, ...]]
    settings_by_plugin: dict[str, AnalyticsPluginExecutionSettings]
    options_by_plugin: dict[str, object | None]
    telemetry: GraphRuntimeTelemetry


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
    telemetry: GraphRuntimeTelemetry | None = None


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
class PluginHashInputs:
    """Immutable inputs for computing plugin hash stability."""

    repo: str
    commit: str
    plugin_name: str
    version_hash: str | None
    scope: GraphRunScope
    options_hash: str | None


def _normalize_options_payload(options: object | None) -> dict[str, object]:
    if options is None:
        return {}
    if isinstance(options, BaseModel):
        return options.model_dump()
    if isinstance(options, dict):
        return options
    message = "Plugin options must be a mapping or BaseModel instance"
    raise TypeError(message)


def _validate_plugin_options(plugin: AnalyticsPlugin, options: dict[str, object]) -> object | None:
    if plugin.options_model is None:
        return options or None
    if not options and plugin.options_default is not None:
        return plugin.options_model.model_validate(
            _normalize_options_payload(plugin.options_default)
        )
    return plugin.options_model.model_validate(options)


def _resolve_analytics_options_map(
    *,
    plugins: Sequence[AnalyticsPlugin],
    cfg_options: Mapping[str, dict[str, object]],
    runtime_options: Mapping[str, dict[str, object]],
) -> dict[str, object | None]:
    allowed_plugins = {plugin.name for plugin in plugins}
    unknown = (set(cfg_options.keys()) | set(runtime_options.keys())) - allowed_plugins
    if unknown:
        message = f"Options provided for unknown analytics plugins: {', '.join(sorted(unknown))}"
        raise ValueError(message)

    resolved: dict[str, object | None] = {}
    for plugin in plugins:
        merged: dict[str, object] = {}
        merged.update(_normalize_options_payload(plugin.options_default))
        merged.update(_normalize_options_payload(cfg_options.get(plugin.name)))
        merged.update(_normalize_options_payload(runtime_options.get(plugin.name)))
        resolved[plugin.name] = _validate_plugin_options(plugin, merged)
    return resolved


def _effective_severity(plugin: AnalyticsPlugin, policy: GraphPluginPolicy) -> str:
    override = policy.severity_overrides.get(plugin.name)
    if override is not None:
        return override
    return plugin.severity if plugin.severity else policy.default_severity


def _effective_timeout(plugin: AnalyticsPlugin, policy: GraphPluginPolicy) -> int | None:
    override = policy.timeouts_ms.get(plugin.name)
    if override is not None:
        return override
    if plugin.resource_hints is None:
        return None
    return plugin.resource_hints.max_runtime_ms


def _compute_options_hash(plugin_name: str, options: object | None) -> str | None:
    if options is None:
        return None
    payload = {"plugin": plugin_name, "options": options}
    return hash_json(payload)


def _compute_input_hash(inputs: PluginHashInputs) -> str:
    scope_payload: dict[str, object] | None = None
    if inputs.scope.paths or inputs.scope.modules or inputs.scope.time_window is not None:
        scope_payload = {
            "paths": inputs.scope.paths,
            "modules": inputs.scope.modules,
            "time_window": (
                (
                    inputs.scope.time_window[0].isoformat(),
                    inputs.scope.time_window[1].isoformat(),
                )
                if inputs.scope.time_window is not None
                else None
            ),
        }
    payload = {
        "repo": inputs.repo,
        "commit": inputs.commit,
        "plugin": inputs.plugin_name,
        "version_hash": inputs.version_hash or "0",
        "options_hash": inputs.options_hash,
        "scope": scope_payload,
    }
    return hash_json(payload)


def _build_analytics_scope(scope: GraphRunScope) -> AnalyticsScope:
    """
    Translate a graph scope into an analytics scope payload.

    Returns
    -------
    AnalyticsScope
        Scope payload aligned to analytics runtime expectations.
    """
    return AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels={"runtime": "analytics"},
    )


def _build_execution_context(
    *,
    plugin: AnalyticsPlugin,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    scratch: GraphRuntimeScratch,
) -> AnalyticsExecutionContext:
    extra_payload = dict(run_context.extra)
    cfgs = run_context.cfgs
    return AnalyticsExecutionContext(
        gateway=run_context.gateway,
        analytics_context=run_context.analytics_context,
        repo=plan.repo,
        commit=plan.commit,
        graph_runtime=run_context.graph_runtime,
        catalog_provider=run_context.catalog_provider,
        function_cfg=cast("FunctionAnalyticsStepConfig | None", cfgs.get("function"))
        if plugin.stage == "function"
        else None,
        function_effects_cfg=cast("FunctionEffectsStepConfig | None", cfgs.get("function_effects"))
        if plugin.name == "functions.effects"
        else None,
        function_contracts_cfg=cast(
            "FunctionContractsStepConfig | None", cfgs.get("function_contracts")
        )
        if plugin.name == "functions.contracts"
        else None,
        function_history_cfg=cast("FunctionHistoryStepConfig | None", cfgs.get("function_history"))
        if plugin.stage == "function_history"
        else None,
        test_profile_cfg=cast("TestProfileStepConfig | None", cfgs.get("test_profile"))
        if plugin.name == "tests.profile" or plugin.stage == "test"
        else None,
        behavioral_cfg=cast("BehavioralCoverageStepConfig | None", cfgs.get("behavioral_coverage"))
        if plugin.name == "tests.behavioral_coverage"
        else None,
        hotspots_cfg=cast("HotspotsStepConfig | None", cfgs.get("hotspots"))
        if plugin.name == "hotspots.build"
        else None,
        subsystems_cfg=cast("SubsystemsStepConfig | None", cfgs.get("subsystems"))
        if plugin.stage == "subsystem"
        else None,
        semantic_roles_cfg=cast("SemanticRolesStepConfig | None", cfgs.get("semantic_roles"))
        if plugin.name == "semantic.roles"
        else None,
        data_models_cfg=cast("DataModelsStepConfig | None", cfgs.get("data_models"))
        if plugin.stage == "data_model"
        else None,
        data_model_usage_cfg=cast("DataModelUsageStepConfig | None", cfgs.get("data_model_usage"))
        if plugin.stage == "data_model_usage"
        else None,
        entrypoints_cfg=cast("EntryPointsStepConfig | None", cfgs.get("entrypoints"))
        if plugin.stage == "entrypoints"
        else None,
        profiles_cfg=cast("ProfilesAnalyticsStepConfig | None", cfgs.get("profiles"))
        if plugin.stage == "profiles"
        else None,
        history_cfg=cast("HistoryTimeseriesStepConfig | None", cfgs.get("history"))
        if plugin.stage == "history"
        else None,
        config_data_flow_cfg=cast("ConfigDataFlowStepConfig | None", cfgs.get("config_data_flow"))
        if plugin.name == "config.data_flow"
        else None,
        coverage_functions_cfg=cast("CoverageAnalyticsStepConfig | None", cfgs.get("coverage_functions"))
        if plugin.name == "coverage.functions"
        else None,
        test_coverage_cfg=cast("TestCoverageStepConfig | None", cfgs.get("test_coverage_edges"))
        if plugin.name == "coverage.test_edges"
        else None,
        external_deps_cfg=cast("ExternalDependenciesStepConfig | None", cfgs.get("external_dependencies"))
        if plugin.name == "deps.external"
        else None,
        graph_cfg=cast("GraphMetricsStepConfig | None", cfgs.get("graph"))
        if plugin.stage == "graph"
        else None,
        options=plan.options_by_plugin.get(plugin.name),
        plugin_name=plugin.name,
        scope=plan.scope,
        run_id=plan.run_id,
        scratch=scratch,
        extra=extra_payload,
    )


def _should_skip_plugin(
    *,
    plan: AnalyticsPluginExecutionPlan,
    plugin: AnalyticsPlugin,
    gateway: StorageGateway,
    settings: AnalyticsPluginExecutionSettings,
) -> tuple[bool, str | None]:
    if plan.policy.dry_run:
        return True, "dry_run"
    if not plan.policy.skip_on_unchanged:
        return False, None
    state = ManifestState(
        plugin_name=plugin.name,
        row_count_tables=plugin.row_count_tables,
        gateway=gateway,
        repo=plan.repo,
        commit=plan.commit,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
    )
    if is_unchanged(plan.prior_manifest, state):
        return True, "unchanged"
    return False, None


def _plugin_row_counts(
    *,
    plugin: AnalyticsPlugin,
    result: object | None,
    run_context: AnalyticsRunContext,
    plan: AnalyticsPluginExecutionPlan,
) -> dict[str, int] | None:
    if isinstance(result, GraphPluginResult):
        return result.row_counts
    return _row_counts_for_tables(
        run_context.gateway,
        repo=plan.repo,
        commit=plan.commit,
        tables=plugin.row_count_tables,
    )


def _analytics_record_from_graph(record: GraphPluginRunRecord) -> AnalyticsRunRecord:
    return AnalyticsRunRecord(
        name=record.name,
        kind="graph_plugin",
        status=record.status,  # type: ignore[arg-type]
        started_at=record.started_at,
        ended_at=record.ended_at,
        duration_ms=record.duration_ms,
        attempts=record.attempts,
        partial=record.partial,
        error=record.error,
        meta=_meta_from_graph_record(record),
    )


def _execute_graph_plugin(
    *,
    plugin: AnalyticsPlugin,
    ctx: GraphMetricExecutionContext,
    plan: AnalyticsPluginExecutionPlan,
) -> tuple[AnalyticsRunRecord, bool]:
    graph_plugin: GraphMetricPlugin = get_graph_metric_plugin(plugin.name)
    settings = plan.settings_by_plugin[plugin.name]
    exec_settings = PluginExecutionSettings(
        name=graph_plugin.name,
        severity=cast("PluginSeverity", settings.severity),
        retry_cfg=settings.retry_cfg,
        timeout_ms=settings.timeout_ms,
        fail_fast=settings.fail_fast,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
        version_hash=settings.version_hash,
        contract_checkers=graph_plugin.contract_checkers,
    )
    span = plan.telemetry.start_plugin(graph_plugin, plan.run_id, ctx)
    try:
        record = _run_graph_execution_plugin(
            plugin=graph_plugin,
            ctx=ctx,
            settings=exec_settings,
            run_id=plan.run_id,
        )
    except PluginFatalError as exc:
        record = exc.record
        plan.telemetry.finish_plugin(span, record)
        plan.telemetry.record_metrics(record, plan.scope)
        return _analytics_record_from_graph(record), True
    plan.telemetry.finish_plugin(span, record)
    plan.telemetry.record_metrics(record, plan.scope)
    stop = record.status == "failed" and exec_settings.severity == "fatal" and exec_settings.fail_fast
    return _analytics_record_from_graph(record), stop


def _execute_graph_plugin_or_skip(
    *,
    plugin: AnalyticsPlugin,
    ctx: GraphMetricExecutionContext,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    settings: AnalyticsPluginExecutionSettings,
) -> tuple[AnalyticsRunRecord, bool]:
    should_skip, skipped_reason = _should_skip_plugin(
        plan=plan,
        plugin=plugin,
        gateway=run_context.gateway,
        settings=settings,
    )
    graph_plugin = get_graph_metric_plugin(plugin.name)
    if should_skip:
        params = RecordParams(
            severity=cast("PluginSeverity", settings.severity),
            timeout_ms=settings.timeout_ms,
            version_hash=settings.version_hash,
            input_hash=settings.input_hash,
            options_hash=settings.options_hash,
            options=plan.options_by_plugin.get(plugin.name),
            requires_isolation=graph_plugin.requires_isolation,
            isolation_kind=graph_plugin.isolation_kind,
            policy_fail_fast=settings.fail_fast,
        )
        if skipped_reason == "dry_run":
            graph_record = dry_run_record(plugin=graph_plugin, params=params, run_id=plan.run_id)
        else:
            graph_record = skip_record(
                plugin=graph_plugin,
                params=params,
                reason=skipped_reason or "skipped",
                run_id=plan.run_id,
            )
        span = plan.telemetry.start_plugin(graph_plugin, plan.run_id, ctx)
        plan.telemetry.finish_plugin(span, graph_record)
        plan.telemetry.record_metrics(graph_record, plan.scope)
        return _analytics_record_from_graph(graph_record), False
    return _execute_graph_plugin(plugin=plugin, ctx=ctx, plan=plan)


def _execute_non_graph_plugin(
    *,
    plugin: AnalyticsPlugin,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    settings: AnalyticsPluginExecutionSettings,
    ctx: AnalyticsExecutionContext,
) -> tuple[AnalyticsRunRecord, bool]:
    started_at = datetime.now(tz=UTC)
    contracts: tuple[PluginContractResult, ...] = ()
    status: str = "skipped"
    attempts = 0
    duration_ms = 0.0
    error: str | None = None
    result: object | None = None
    skipped_reason: str | None = None

    should_skip, skipped_reason = _should_skip_plugin(
        plan=plan,
        plugin=plugin,
        gateway=run_context.gateway,
        settings=settings,
    )
    if not should_skip:
        status, error, duration_ms, attempts, result = _execute_with_retries(
            plugin=plugin,
            ctx=ctx,
            settings=settings,
        )
        if status == "succeeded" and plugin.contract_checkers:
            contracts = run_contract_checkers(
                ctx=ctx,  # type: ignore[arg-type]
                checkers=plugin.contract_checkers,
            )
        if status == "skipped" and settings.severity == "skip_on_error":
            skipped_reason = "skip_on_error"

    ended_at = datetime.now(tz=UTC)
    row_counts = _plugin_row_counts(
        plugin=plugin,
        result=result,
        run_context=run_context,
        plan=plan,
    )
    record = AnalyticsRunRecord(
        name=plugin.name,
        kind=plugin.stage,
        status=status,  # type: ignore[arg-type]
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        attempts=attempts,
        partial=status != "succeeded",
        error=error,
        meta={
            "stage": plugin.stage,
            "severity": settings.severity,
            "options_hash": settings.options_hash,
            "version_hash": settings.version_hash,
            "input_hash": settings.input_hash,
            "row_counts": row_counts,
            "result": result,
            "timeout_ms": settings.timeout_ms,
            "contracts": contracts,
            "skipped_reason": skipped_reason,
            "policy_fail_fast": settings.fail_fast,
            "requires_isolation": plugin.requires_isolation,
            "isolation_kind": plugin.isolation_kind,
        },
    )
    stop = status == "failed" and settings.severity == "fatal" and settings.fail_fast
    return record, stop


def _execute_plugin(
    *,
    plugin: AnalyticsPlugin,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
    scratch: GraphRuntimeScratch,
) -> tuple[AnalyticsRunRecord, bool]:
    settings = plan.settings_by_plugin[plugin.name]
    ctx = _build_execution_context(
        plugin=plugin,
        plan=plan,
        run_context=run_context,
        scratch=scratch,
    )
    if plugin.context_factory is not None:
        ctx = plugin.context_factory(ctx)  # type: ignore[assignment]
    if isinstance(ctx, GraphMetricExecutionContext):
        return _execute_graph_plugin_or_skip(
            plugin=plugin,
            ctx=ctx,
            plan=plan,
            run_context=run_context,
            settings=settings,
        )
    return _execute_non_graph_plugin(
        plugin=plugin,
        plan=plan,
        run_context=run_context,
        settings=settings,
        ctx=ctx,
    )


def plan_analytics_plugin_run(request: AnalyticsPlanRequest) -> AnalyticsPluginExecutionPlan:
    """
    Plan an analytics plugin execution run.

    Parameters
    ----------
    request
        Complete planning inputs including policy, scope, and plugin names.

    Returns
    -------
    AnalyticsPluginExecutionPlan
        Ordered plugin plan with resolved options and execution settings.
    """
    plan = plan_analytics_plugins(request.plugin_names)
    plugins = plan.plugins

    options_by_plugin = _resolve_analytics_options_map(
        plugins=plugins,
        cfg_options=request.cfg_options or {},
        runtime_options=request.runtime_options or {},
    )

    settings_by_plugin: dict[str, AnalyticsPluginExecutionSettings] = {}
    telemetry = request.telemetry or NoOpGraphRuntimeTelemetry()
    for plugin in plugins:
        options = options_by_plugin.get(plugin.name)
        options_hash = _compute_options_hash(plugin.name, options)
        severity = _effective_severity(plugin, request.policy)
        retry_cfg = request.policy.retries.get(plugin.name, GraphPluginRetryPolicy())
        timeout_ms = _effective_timeout(plugin, request.policy)
        input_hash = _compute_input_hash(
            PluginHashInputs(
                repo=request.repo,
                commit=request.commit,
                plugin_name=plugin.name,
                version_hash=plugin.version_hash,
                scope=request.scope,
                options_hash=options_hash,
            )
        )
        settings_by_plugin[plugin.name] = AnalyticsPluginExecutionSettings(
            name=plugin.name,
            severity=severity,
            retry_cfg=retry_cfg,
            timeout_ms=timeout_ms,
            fail_fast=request.policy.fail_fast,
            input_hash=input_hash,
            options_hash=options_hash,
            version_hash=plugin.version_hash,
        )

    skipped = tuple(
        AnalyticsSkippedStep(name=s.name, reason=s.reason, kind=s.kind)
        for s in plan.skipped_plugins
    )

    return AnalyticsPluginExecutionPlan(
        plan_id=plan.plan_id,
        run_id=request.run_id,
        repo=request.repo,
        commit=request.commit,
        policy=request.policy,
        prior_manifest=request.prior_manifest,
        scope=request.scope,
        plugins=plugins,
        ordered_names=plan.ordered_names,
        skipped=skipped,
        dep_graph=plan.dep_graph,
        settings_by_plugin=settings_by_plugin,
        options_by_plugin=dict(options_by_plugin),
        telemetry=telemetry,
    )


def _row_counts_for_tables(
    gateway: StorageGateway | None,
    *,
    repo: str,
    commit: str,
    tables: Sequence[str],
) -> dict[str, int] | None:
    if gateway is None or not tables:
        return None
    connection = getattr(gateway, "con", None)
    if connection is None:
        return None
    counts: dict[str, int] = {}
    for table in tables:
        try:
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = connection.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            counts[table] = int(relation.count().fetchone()[0])
        except Exception:  # noqa: BLE001
            log.debug("row_count.failed table=%s repo=%s commit=%s", table, repo, commit)
            return None
    return counts


def _execute_with_retries(
    *,
    plugin: AnalyticsPlugin,
    ctx: AnalyticsExecutionContext,
    settings: AnalyticsPluginExecutionSettings,
) -> tuple[str, str | None, float, int, object | None]:
    start = time.perf_counter()
    attempts = 0
    error: str | None = None
    result: object | None = None
    status = "succeeded"

    max_attempts = max(settings.retry_cfg.max_attempts, 1)
    while attempts < max_attempts:
        attempts += 1
        try:
            result = plugin.run(ctx)  # type: ignore[arg-type]
            status = "succeeded"
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if settings.severity == "skip_on_error":
                status = "skipped"
                break
            if attempts >= max_attempts:
                status = "failed"
                break
            if settings.retry_cfg.backoff_ms > 0:
                time.sleep(settings.retry_cfg.backoff_ms / 1000)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    return status, error, duration_ms, attempts, result


def run_analytics_plugins(
    *,
    plan: AnalyticsPluginExecutionPlan,
    run_context: AnalyticsRunContext,
) -> AnalyticsRunReport:
    """
    Execute all plugins in `plan` using a shared harness.

    Parameters
    ----------
    plan
        Planned plugins with settings and dependency ordering.
    run_context
        Shared runtime context (gateway, runtimes, configs) for execution.

    Returns
    -------
    AnalyticsRunReport
        Telemetry and records for each executed plugin.
    """
    records: list[AnalyticsRunRecord] = []
    scratch = GraphRuntimeScratch()

    analytics_scope = _build_analytics_scope(plan.scope)

    for plugin in plan.plugins:
        record, stop = _execute_plugin(
            plugin=plugin,
            plan=plan,
            run_context=run_context,
            scratch=scratch,
        )
        records.append(record)
        if stop:
            break

    scratch.cleanup()

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


__all__ = [
    "AnalyticsPlanRequest",
    "AnalyticsPluginExecutionPlan",
    "AnalyticsPluginExecutionSettings",
    "AnalyticsPluginRunOptions",
    "AnalyticsRunContext",
    "plan_analytics_plugin_run",
    "run_analytics_plugins",
]
