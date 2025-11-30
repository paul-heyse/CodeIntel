"""Generic analytics plugin planning and execution harness."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.contracts import PluginContractResult, run_contract_checkers
from codeintel.analytics.graphs.plugins import GraphPluginResult, GraphRuntimeScratch
from codeintel.analytics.graphs.runtime.manifest import ManifestState, hash_json, is_unchanged
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    AnalyticsPluginPlan,
    plan_analytics_plugins,
)
from codeintel.analytics.runtime_manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
)
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.storage.gateway import StorageGateway

try:  # pragma: no cover - optional dependency at runtime
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
except Exception:  # pragma: no cover - fallback for type checkers
    FunctionCatalogProvider = None  # type: ignore[assignment]

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
        return plugin.options_model.model_validate(_normalize_options_payload(plugin.options_default))
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
        message = (
            "Options provided for unknown analytics plugins: "
            f"{', '.join(sorted(unknown))}"
        )
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


def _compute_input_hash(
    *,
    repo: str,
    commit: str,
    plugin_name: str,
    version_hash: str | None,
    scope: GraphRunScope,
    options_hash: str | None,
) -> str:
    scope_payload: dict[str, object] | None = None
    if scope.paths or scope.modules or scope.time_window is not None:
        scope_payload = {
            "paths": scope.paths,
            "modules": scope.modules,
            "time_window": (
                (
                    scope.time_window[0].isoformat(),
                    scope.time_window[1].isoformat(),
                )
                if scope.time_window is not None
                else None
            ),
        }
    payload = {
        "repo": repo,
        "commit": commit,
        "plugin": plugin_name,
        "version_hash": version_hash or "0",
        "options_hash": options_hash,
        "scope": scope_payload,
    }
    return hash_json(payload)


def plan_analytics_plugin_run(
    plugin_names: Sequence[str],
    *,
    policy: GraphPluginPolicy,
    repo: str,
    commit: str,
    scope: GraphRunScope,
    prior_manifest: Mapping[str, Mapping[str, object]] | None,
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
    run_id: str,
) -> AnalyticsPluginExecutionPlan:
    """Generic plugin planning."""

    plan = plan_analytics_plugins(plugin_names)
    plugins = plan.plugins

    options_by_plugin = _resolve_analytics_options_map(
        plugins=plugins,
        cfg_options=cfg_options or {},
        runtime_options=runtime_options or {},
    )

    settings_by_plugin: dict[str, AnalyticsPluginExecutionSettings] = {}
    for plugin in plugins:
        options = options_by_plugin.get(plugin.name)
        options_hash = _compute_options_hash(plugin.name, options)
        severity = _effective_severity(plugin, policy)
        retry_cfg = policy.retries.get(plugin.name, GraphPluginRetryPolicy())
        timeout_ms = _effective_timeout(plugin, policy)
        input_hash = _compute_input_hash(
            repo=repo,
            commit=commit,
            plugin_name=plugin.name,
            version_hash=plugin.version_hash,
            scope=scope,
            options_hash=options_hash,
        )
        settings_by_plugin[plugin.name] = AnalyticsPluginExecutionSettings(
            name=plugin.name,
            severity=severity,
            retry_cfg=retry_cfg,
            timeout_ms=timeout_ms,
            fail_fast=policy.fail_fast,
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
        run_id=run_id,
        repo=repo,
        commit=commit,
        policy=policy,
        prior_manifest=prior_manifest,
        scope=scope,
        plugins=plugins,
        ordered_names=plan.ordered_names,
        skipped=skipped,
        dep_graph=plan.dep_graph,
        settings_by_plugin=settings_by_plugin,
        options_by_plugin=dict(options_by_plugin),
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
    gateway: StorageGateway,
    analytics_context: AnalyticsContext | None,
    graph_runtime: GraphRuntime | None,
    cfgs: dict[str, object],
    extra: dict[str, Any] | None = None,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> AnalyticsRunReport:
    """Execute all plugins in `plan` using a shared harness."""

    records: list[AnalyticsRunRecord] = []
    scratch = GraphRuntimeScratch()

    scope = plan.scope
    analytics_scope = AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels={"runtime": "analytics"},
    )

    extra_payload = extra or {}

    for plugin in plan.plugins:
        settings = plan.settings_by_plugin[plugin.name]
        options = plan.options_by_plugin.get(plugin.name)
        ctx = AnalyticsExecutionContext(
            gateway=gateway,
            analytics_context=analytics_context,
            repo=plan.repo,
            commit=plan.commit,
            graph_runtime=graph_runtime if plugin.stage == "graph" else None,
            catalog_provider=catalog_provider if plugin.stage == "graph" else None,
            function_cfg=cfgs.get("function") if plugin.stage == "function" else None,
            function_effects_cfg=cfgs.get("function_effects")
            if plugin.name == "functions.effects"
            else None,
            function_contracts_cfg=cfgs.get("function_contracts")
            if plugin.name == "functions.contracts"
            else None,
            function_history_cfg=cfgs.get("function_history")
            if plugin.stage == "function_history"
            else None,
            test_profile_cfg=cfgs.get("test_profile")
            if plugin.name == "tests.profile" or plugin.stage == "test"
            else None,
            behavioral_cfg=cfgs.get("behavioral_coverage")
            if plugin.name == "tests.behavioral_coverage"
            else None,
            hotspots_cfg=cfgs.get("hotspots") if plugin.name == "hotspots.build" else None,
            subsystems_cfg=cfgs.get("subsystems") if plugin.stage == "subsystem" else None,
            semantic_roles_cfg=cfgs.get("semantic_roles")
            if plugin.name == "semantic.roles"
            else None,
            data_models_cfg=cfgs.get("data_models") if plugin.stage == "data_model" else None,
            data_model_usage_cfg=cfgs.get("data_model_usage")
            if plugin.stage == "data_model_usage"
            else None,
            entrypoints_cfg=cfgs.get("entrypoints") if plugin.stage == "entrypoints" else None,
            profiles_cfg=cfgs.get("profiles") if plugin.stage == "profiles" else None,
            history_cfg=cfgs.get("history") if plugin.stage == "history" else None,
            config_data_flow_cfg=cfgs.get("config_data_flow")
            if plugin.name == "config.data_flow"
            else None,
            coverage_functions_cfg=cfgs.get("coverage_functions")
            if plugin.name == "coverage.functions"
            else None,
            test_coverage_cfg=cfgs.get("test_coverage_edges")
            if plugin.name == "coverage.test_edges"
            else None,
            external_deps_cfg=cfgs.get("external_dependencies")
            if plugin.name == "deps.external"
            else None,
            graph_cfg=cfgs.get("graph") if plugin.stage == "graph" else None,
            options=options,
            plugin_name=plugin.name,
            scope=plan.scope,
            run_id=plan.run_id,
            scratch=scratch,
            extra=dict(extra_payload),
        )

        if plugin.context_factory is not None:
            ctx = plugin.context_factory(ctx)  # type: ignore[assignment]

        started_at = datetime.now(tz=UTC)
        attempts = 0
        duration_ms = 0.0
        error: str | None = None
        result: object | None = None
        contracts: tuple[PluginContractResult, ...] = ()
        skipped_reason: str | None = None

        if plan.policy.dry_run:
            status = "skipped"
            attempts = 0
            skipped_reason = "dry_run"
        else:
            unchanged = False
            if plan.policy.skip_on_unchanged:
                state = ManifestState(
                    plugin_name=plugin.name,
                    row_count_tables=plugin.row_count_tables,
                    gateway=gateway,
                    repo=plan.repo,
                    commit=plan.commit,
                    input_hash=settings.input_hash,
                    options_hash=settings.options_hash,
                )
                unchanged = is_unchanged(plan.prior_manifest, state)
            if unchanged:
                status = "skipped"
                attempts = 0
                error = None
                duration_ms = 0.0
                skipped_reason = "unchanged"
            else:
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

        plugin_row_counts = (
            result.row_counts if isinstance(result, GraphPluginResult) else None
        )
        row_counts = plugin_row_counts or _row_counts_for_tables(
            gateway,
            repo=plan.repo,
            commit=plan.commit,
            tables=plugin.row_count_tables,
        )
        records.append(
            AnalyticsRunRecord(
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
                },
            )
        )

        if (
            status == "failed"
            and settings.severity == "fatal"
            and settings.fail_fast
        ):
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
    "AnalyticsPluginExecutionPlan",
    "AnalyticsPluginExecutionSettings",
    "AnalyticsPluginRunOptions",
    "plan_analytics_plugin_run",
    "run_analytics_plugins",
]
