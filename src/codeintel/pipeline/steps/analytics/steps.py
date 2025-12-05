"""Analytics-focused pipeline steps."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeintel.analytics.core.pipeline_bridge import (
    AnalyticsPlanRequest,
    AnalyticsRunContext,
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.analytics.plugins import (
    BEHAVIORAL_COVERAGE_PLUGIN,
    CONFIG_DATA_FLOW_PLUGIN,
    COVERAGE_FUNCTIONS_PLUGIN,
    COVERAGE_TEST_EDGES_PLUGIN,
    DATA_MODEL_USAGE_PLUGIN,
    DATA_MODELS_PLUGIN,
    ENTRYPOINTS_PLUGIN,
    EXTERNAL_DEPS_PLUGIN,
    FUNCTION_AST_FEATURES_PLUGIN,
    FUNCTION_CONTRACTS_PLUGIN,
    FUNCTION_EFFECTS_PLUGIN,
    FUNCTION_HISTORY_PLUGIN,
    FUNCTION_METRICS_PLUGIN,
    HISTORY_TIMESERIES_PLUGIN,
    HOTSPOTS_PLUGIN,
    PROFILES_PLUGIN,
    RISK_FACTORS_PLUGIN,
    SEMANTIC_ROLES_PLUGIN,
    SUBSYSTEMS_PLUGIN,
    TEST_PROFILE_PLUGIN,
    ensure_plugins_registered,
)
from codeintel.analytics.runtime.context import load_prior_manifest
from codeintel.analytics.runtime.manifest import encode_manifest
from codeintel.analytics.subsystems import refresh_subsystem_caches
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.graphs.catalog import FunctionCatalogProvider
from codeintel.graphs.core.protocol import DEFAULT_METRIC_PLUGINS
from codeintel.graphs.recipes import METRICS_ONLY_RECIPE, RecipeExecutor, RecipeExecutorContext
from codeintel.pipeline.execution.context import (
    PipelineContext,
    _function_catalog,
    _log_step,
    _resolve_code_profile,
    ensure_graph_runtime,
)
from codeintel.pipeline.steps.base import PipelineStep, StepPhase
from codeintel.storage.gateway import StorageGateway, build_snapshot_gateway_resolver

log = logging.getLogger(__name__)

# Ensure plugins are registered on module import
ensure_plugins_registered()


def _config_data_flow_plugin_name() -> str:
    """Return the config data flow plugin name.

    Returns
    -------
    str
        Plugin name for config data flow.
    """
    return CONFIG_DATA_FLOW_PLUGIN.metadata.name


def _parse_commits(commits_extra: object, commits_env: str) -> tuple[str, ...]:
    """
    Normalize commit configuration from env vars and pipeline extras.

    Returns
    -------
    tuple[str, ...]
        Ordered commit identifiers with duplicates removed.
    """
    commits_from_env = tuple(commit for commit in commits_env.split(",") if commit)
    if isinstance(commits_extra, str):
        commits_from_extra = tuple(commit for commit in commits_extra.split(",") if commit)
    elif isinstance(commits_extra, Iterable):
        commits_from_extra = tuple(str(commit) for commit in commits_extra)
    else:
        commits_from_extra = ()
    return tuple(commit for commit in (*commits_from_extra, *commits_from_env) if commit)


def _seed_catalog_modules(
    gateway: StorageGateway,
    catalog: FunctionCatalogProvider | None,
    *,
    repo: str,
    commit: str,
) -> bool:
    """
    Create a temporary table of modules from a catalog when available.

    Returns
    -------
    bool
        True when a temp table was created.
    """
    if catalog is None:
        return False
    if not hasattr(catalog, "catalog"):
        return False
    module_by_path = catalog.catalog().module_by_path
    if not module_by_path:
        return False
    con = gateway.con
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE temp.catalog_modules (
            path VARCHAR,
            module VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            tags JSON,
            owners JSON
        )
        """
    )
    con.executemany(
        "INSERT INTO temp.catalog_modules VALUES (?, ?, ?, ?, ?, ?)",
        [(path, module, repo, commit, "[]", "[]") for path, module in module_by_path.items()],
    )
    return True


@dataclass
class HotspotsStep:
    """Build analytics.hotspots from core.ast_metrics plus git churn."""

    name: str = "hotspots"
    description: str = "Compute file-level hotspot scores from AST metrics and git churn."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("ast_extract",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute file-level hotspot scores."""
        _log_step(self.name)
        cfg = ctx.config_builder().hotspots()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "hotspots.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(HOTSPOTS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"hotspots": cfg},
                extra={"tool_runner": ctx.tool_runner},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class FunctionHistoryStep:
    """Aggregate per-function git history."""

    name: str = "function_history"
    description: str = "Aggregate git churn and commit history per function GOID."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("function_metrics", "hotspots")

    def run(self, ctx: PipelineContext) -> None:
        """Compute git churn and history for each function GOID."""
        _log_step(self.name)
        cfg = ctx.config_builder().function_history()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "function_history.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(FUNCTION_HISTORY_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"function_history": cfg},
                extra={"tool_runner": ctx.tool_runner},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class HistoryTimeseriesStep:
    """Aggregate cross-commit analytics.history_timeseries."""

    name: str = "history_timeseries"
    description: str = "Aggregate analytics across commits into history timeseries."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("profiles",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute history timeseries when commit list is provided."""
        _log_step(self.name)
        commits_env = os.getenv("CODEINTEL_HISTORY_COMMITS", "")
        commits_extra = ctx.extra.get("history_commits")
        commits_raw = _parse_commits(commits_extra, commits_env)
        commits = commits_raw if ctx.commit in commits_raw else (ctx.commit, *commits_raw)
        if not commits:
            log.info("Skipping history_timeseries: no commits configured.")
            return

        db_dir_env = os.getenv("CODEINTEL_HISTORY_DB_DIR")
        history_db_dir = Path(db_dir_env) if db_dir_env else ctx.build_dir / "db"
        history_db_dir.mkdir(parents=True, exist_ok=True)

        cfg = ctx.config_builder().history_timeseries(commits=commits)
        snapshot_resolver = build_snapshot_gateway_resolver(
            db_dir=history_db_dir,
            repo=ctx.repo,
            primary_gateway=ctx.gateway,
        )
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "history_timeseries.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(HISTORY_TIMESERIES_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"history": cfg},
                extra={
                    "history_snapshot_resolver": snapshot_resolver,
                    "tool_runner": ctx.tool_runner,
                },
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types."""

    name: str = "function_metrics"
    description: str = "Compute per-function metrics, complexity, and type annotations."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute per-function metrics and typedness via AnalyticsPlugin harness."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().function_analytics(
            fail_on_missing_spans=ctx.function_fail_on_missing_spans,
            parser=ctx.function_parser,
        )
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "function_metrics.json"
        prior_manifest = load_prior_manifest(manifest_path)

        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(
                    FUNCTION_AST_FEATURES_PLUGIN.metadata.name,
                    FUNCTION_METRICS_PLUGIN.metadata.name,
                ),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )

        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"function": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )

        summary: dict[str, int] = {
            "metrics_rows": 0,
            "types_rows": 0,
            "validation_total": 0,
            "validation_parse_failed": 0,
            "validation_span_not_found": 0,
            "rows_written": 0,
            "functions_seen": 0,
            "functions_missing": 0,
        }
        for rec in report.records:
            if rec.name == "functions.metrics" and isinstance(rec.meta, dict):
                result = rec.meta.get("result")
                if isinstance(result, dict):
                    summary.update({k: int(v) for k, v in result.items() if isinstance(v, int)})
            if rec.name == "functions.ast_features" and isinstance(rec.meta, dict):
                result = rec.meta.get("result")
                if isinstance(result, dict):
                    summary.update({k: int(v) for k, v in result.items() if isinstance(v, int)})

        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log.info(
            (
                "function_metrics summary rows=%d types=%d validation=%d "
                "parse_failed=%d span_not_found=%d ast_features=%d functions_seen=%d missing=%d"
            ),
            summary["metrics_rows"],
            summary["types_rows"],
            summary["validation_total"],
            summary["validation_parse_failed"],
            summary["validation_span_not_found"],
            summary["rows_written"],
            summary["functions_seen"],
            summary["functions_missing"],
        )


@dataclass
class FunctionEffectsStep:
    """Classify side effects and purity for functions."""

    name: str = "function_effects"
    description: str = "Classify side effects and purity for each function."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids", "callgraph")

    def run(self, ctx: PipelineContext) -> None:
        """Compute function_effects flags and evidence."""
        _log_step(self.name)
        cfg = ctx.config_builder().function_effects()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "function_effects.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(FUNCTION_EFFECTS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"function_effects": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class FunctionContractsStep:
    """Infer pre/postconditions and nullability."""

    name: str = "function_contracts"
    description: str = "Infer pre/postconditions and nullability contracts for functions."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("function_metrics", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Compute inferred contracts for functions."""
        _log_step(self.name)
        cfg = ctx.config_builder().function_contracts()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "function_contracts.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(FUNCTION_CONTRACTS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"function_contracts": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class DataModelsStep:
    """Extract structured data models from class definitions."""

    name: str = "data_models"
    description: str = "Extract structured data models from class definitions."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("ast_extract", "goids", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.data_models."""
        _log_step(self.name)
        cfg = ctx.config_builder().data_models()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "data_models.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(DATA_MODELS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"data_models": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class DataModelUsageStep:
    """Classify per-function data model usage."""

    name: str = "data_model_usage"
    description: str = "Classify per-function data model read/write usage patterns."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("data_models", "callgraph", "cfg", "function_metrics")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.data_model_usage."""
        _log_step(self.name)
        cfg = ctx.config_builder().data_model_usage()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "data_model_usage.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(DATA_MODEL_USAGE_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"data_model_usage": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ConfigDataFlowStep:
    """Track config key usage at the function level."""

    name: str = "config_data_flow"
    description: str = "Track configuration key usage and data flow at the function level."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("config_ingest", "callgraph", "function_metrics", "entrypoints")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.config_data_flow."""
        _log_step(self.name)
        cfg = ctx.config_builder().config_data_flow()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "config_data_flow.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plugin_name = _config_data_flow_plugin_name()
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(plugin_name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"config_data_flow": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class CoverageAnalyticsStep:
    """Build analytics.coverage_functions from GOIDs and coverage_lines."""

    name: str = "coverage_functions"
    description: str = "Aggregate line coverage data to function-level metrics."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids", "coverage_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Aggregate line coverage to function spans."""
        _log_step(self.name)
        cfg = ctx.config_builder().coverage_analytics()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "coverage_functions.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(COVERAGE_FUNCTIONS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"coverage_functions": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class TestCoverageEdgesStep:
    """Build analytics.test_coverage_edges from coverage contexts."""

    name: str = "test_coverage_edges"
    description: str = "Build test-to-function coverage edges from coverage contexts."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("coverage_ingest", "tests_ingest", "goids")

    def run(self, ctx: PipelineContext) -> None:
        """Derive test-to-function edges using coverage contexts."""
        _log_step(self.name)
        cfg = ctx.config_builder().test_coverage(coverage_loader=ctx.coverage_loader)
        catalog = _function_catalog(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "test_coverage_edges.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(COVERAGE_TEST_EDGES_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"test_coverage_edges": cfg},
                extra={},
                catalog_provider=catalog,
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class RiskFactorsStep:
    """Aggregate analytics into analytics.goid_risk_factors."""

    name: str = "risk_factors"
    description: str = "Aggregate analytics into per-function risk scores and levels."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = (
        "function_metrics",
        "coverage_functions",
        "hotspots",
        "typing_ingest",
        "tests_ingest",
        "test_coverage_edges",
        "config_ingest",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Compute risk factors by joining analytics tables."""
        _log_step(self.name)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "risk_factors.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(RISK_FACTORS_PLUGIN.metadata.name,),
                policy=policy,
                repo=ctx.repo,
                commit=ctx.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_graph_plugins(
    cfg: GraphMetricsStepConfig,
    default_plugins: Sequence[str],
) -> tuple[str, ...]:
    """Resolve effective graph metric plugins from config and defaults.

    This function is used for logging purposes only. Actual execution
    uses the recipe executor which handles plugin orchestration.

    Rules
    -----
    - If cfg.enabled_plugins is non-empty, use that list exactly (in order).
    - Otherwise, start from default_plugins and drop any in cfg.disabled_plugins.

    Returns
    -------
    tuple[str, ...]
        Ordered plugin names to execute.
    """
    # Simple enable/disable logic without full dependency resolution
    # since actual execution happens via RecipeExecutor
    if cfg.enabled_plugins:
        return tuple(cfg.enabled_plugins)

    disabled = frozenset(cfg.disabled_plugins) if cfg.disabled_plugins else frozenset()
    return tuple(p for p in default_plugins if p not in disabled)


@dataclass
class GraphMetricsStep:
    """Compute graph metrics for functions and modules."""

    name: str = "graph_metrics"
    description: str = "Compute centrality, coupling, and graph metrics for functions and modules."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("callgraph", "import_graph", "symbol_uses", "cfg", "test_coverage_edges")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.graph_metrics_* tables."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg_scope = ctx.graph_scope
        cfg = ctx.config_builder().graph_metrics(scope=cfg_scope)
        runtime = ensure_graph_runtime(ctx)

        plugin_names = _resolve_graph_plugins(cfg, DEFAULT_METRIC_PLUGINS)
        log.info(
            "graph_metrics.plugins repo=%s commit=%s plugins=%s",
            ctx.repo,
            ctx.commit,
            plugin_names,
        )

        # Use the new RecipeExecutor for graph metrics
        executor_ctx = RecipeExecutorContext(
            gateway=gateway,
            snapshot=ctx.snapshot,
            engine=runtime.engine,
            catalog_provider=_function_catalog(ctx),
        )
        executor = RecipeExecutor(executor_ctx)
        result = executor.execute(METRICS_ONLY_RECIPE)

        log.info(
            "graph_metrics.complete repo=%s commit=%s success=%s "
            "succeeded=%d failed=%d skipped=%d duration_ms=%.2f",
            ctx.repo,
            ctx.commit,
            result.success,
            result.success_count,
            result.failure_count,
            result.skip_count,
            result.duration_ms,
        )


@dataclass
class SemanticRolesStep:
    """Classify functions and modules into semantic roles."""

    name: str = "semantic_roles"
    description: str = (
        "Classify functions and modules into semantic roles like handler, service, util."
    )
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = (
        "function_effects",
        "function_contracts",
        "graph_metrics",
        "function_metrics",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Compute semantic role tables."""
        _log_step(self.name)
        cfg = ctx.config_builder().semantic_roles()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "semantic_roles.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(SEMANTIC_ROLES_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"semantic_roles": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class SubsystemsStep:
    """Infer subsystems from module coupling and risk signals."""

    name: str = "subsystems"
    description: str = "Infer subsystems from module coupling and risk signals."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("import_graph", "symbol_uses", "risk_factors")

    def run(self, ctx: PipelineContext) -> None:
        """Populate subsystem membership and summaries."""
        _log_step(self.name)
        cfg = ctx.config_builder().subsystems()
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "subsystems.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(SUBSYSTEMS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"subsystems": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class TestProfileStep:
    """Build per-test profiles."""

    name: str = "test_profile"
    description: str = "Build per-test profiles with coverage and subsystem context."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = (
        "tests_ingest",
        "coverage_functions",
        "test_coverage_edges",
        "subsystems",
        "graph_metrics",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.test_profile."""
        _log_step(self.name)
        cfg = ctx.config_builder().test_profile()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "test_profile.json"
        prior_manifest = load_prior_manifest(manifest_path)

        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(TEST_PROFILE_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )

        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"test_profile": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )

        summary = {}
        for rec in report.records:
            if rec.name == "tests.profile" and isinstance(rec.meta, dict):
                if isinstance(rec.meta.get("result"), dict):
                    summary = rec.meta["result"]
                break

        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log.info("test_profile summary rows=%d", summary.get("profile_rows", 0))
        if cfg.refresh_subsystem_cache:
            refresh_subsystem_caches(
                ctx.gateway,
                repo=cfg.repo,
                commit=cfg.commit,
                benchmark=cfg.benchmark_subsystem_cache,
            )


@dataclass
class BehavioralCoverageStep:
    """Assign heuristic behavior tags to tests."""

    name: str = "behavioral_coverage"
    description: str = "Assign heuristic behavior tags to tests (unit, integration, etc.)."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("test_profile",)

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.behavioral_coverage."""
        _log_step(self.name)
        enable_llm = bool(
            ctx.extra.get("enable_behavioral_llm")
            or os.getenv("CODEINTEL_BEHAVIORAL_LLM", "").lower() in {"1", "true", "yes"}
        )
        llm_model_raw = ctx.extra.get("behavioral_llm_model")
        llm_model = llm_model_raw if isinstance(llm_model_raw, str) else None
        llm_runner = ctx.extra.get("behavioral_llm_runner")
        cfg = ctx.config_builder().behavioral_coverage(
            enable_llm=enable_llm,
            llm_model=llm_model,
        )
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "behavioral_coverage.json"
        prior_manifest = load_prior_manifest(manifest_path)

        cfg_options: dict[str, dict[str, object]] = {
            BEHAVIORAL_COVERAGE_PLUGIN.metadata.name: {
                "enable_llm": cfg.enable_llm,
                "llm_model": cfg.llm_model,
            }
        }

        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(BEHAVIORAL_COVERAGE_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options=cfg_options,
                runtime_options={},
                run_id=ctx.run_id,
            )
        )

        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"behavioral_coverage": cfg},
                extra={"behavioral_llm_runner": llm_runner},
                catalog_provider=_function_catalog(ctx),
            ),
        )

        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summary = {}
        for rec in report.records:
            if rec.name == "tests.behavioral_coverage" and isinstance(rec.meta, dict):
                if isinstance(rec.meta.get("result"), dict):
                    summary = rec.meta["result"]
                break
        log.info(
            "behavioral_coverage summary rows=%d enable_llm=%s llm_model=%s",
            summary.get("behavior_rows", 0),
            cfg.enable_llm,
            cfg.llm_model,
        )


@dataclass
class EntryPointsStep:
    """Detect HTTP/CLI/job entrypoints and map them to handlers and tests."""

    name: str = "entrypoints"
    description: str = "Detect HTTP/CLI/job entrypoints and map them to handlers and tests."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = (
        "subsystems",
        "coverage_functions",
        "test_coverage_edges",
        "goids",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.entrypoints and analytics.entrypoint_tests."""
        _log_step(self.name)
        cfg = ctx.config_builder().entrypoints(scan_profile=_resolve_code_profile(ctx))
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "entrypoints.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(ENTRYPOINTS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"entrypoints": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ExternalDependenciesStep:
    """Identify external dependency usage across functions."""

    name: str = "external_dependencies"
    description: str = "Identify external dependency usage across functions."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids", "config_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate dependency call edges and aggregated usage."""
        _log_step(self.name)
        cfg = ctx.config_builder().external_dependencies(
            scan_profile=_resolve_code_profile(ctx),
        )
        runtime = ensure_graph_runtime(ctx)
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "external_dependencies.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(EXTERNAL_DEPS_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=runtime,
                cfgs={"external_dependencies": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ProfilesStep:
    """Build function, file, and module profiles."""

    name: str = "profiles"
    description: str = "Build aggregated profiles for functions, files, and modules."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = (
        "risk_factors",
        "callgraph",
        "import_graph",
        "function_effects",
        "function_contracts",
        "semantic_roles",
        "function_history",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Aggregate profile tables for functions, files, and modules."""
        _log_step(self.name)
        cfg = ctx.config_builder().profiles_analytics()
        policy = GraphPluginPolicy()
        scope = GraphRunScope()
        manifest_path = ctx.build_dir / "manifests" / "profiles.json"
        prior_manifest = load_prior_manifest(manifest_path)
        plan = plan_analytics_plugin_run(
            AnalyticsPlanRequest(
                plugin_names=(PROFILES_PLUGIN.metadata.name,),
                policy=policy,
                repo=cfg.repo,
                commit=cfg.commit,
                scope=scope,
                prior_manifest=prior_manifest or {},
                cfg_options={},
                runtime_options={},
                run_id=ctx.run_id,
            )
        )
        report = run_analytics_plugins(
            plan=plan,
            run_context=AnalyticsRunContext(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                graph_runtime=None,
                cfgs={"profiles": cfg},
                extra={},
                catalog_provider=_function_catalog(ctx),
            ),
        )
        if manifest_path is not None:
            payload = encode_manifest(report)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


ANALYTICS_STEPS: dict[str, PipelineStep] = {
    "hotspots": HotspotsStep(),
    "function_history": FunctionHistoryStep(),
    "function_metrics": FunctionAnalyticsStep(),
    "function_effects": FunctionEffectsStep(),
    "function_contracts": FunctionContractsStep(),
    "data_models": DataModelsStep(),
    "data_model_usage": DataModelUsageStep(),
    "config_data_flow": ConfigDataFlowStep(),
    "coverage_functions": CoverageAnalyticsStep(),
    "test_coverage_edges": TestCoverageEdgesStep(),
    "risk_factors": RiskFactorsStep(),
    "graph_metrics": GraphMetricsStep(),
    "subsystems": SubsystemsStep(),
    "semantic_roles": SemanticRolesStep(),
    "entrypoints": EntryPointsStep(),
    "external_dependencies": ExternalDependenciesStep(),
    "test_profile": TestProfileStep(),
    "behavioral_coverage": BehavioralCoverageStep(),
    "profiles": ProfilesStep(),
    "history_timeseries": HistoryTimeseriesStep(),
}


__all__ = [
    "ANALYTICS_STEPS",
    "BehavioralCoverageStep",
    "ConfigDataFlowStep",
    "CoverageAnalyticsStep",
    "DataModelUsageStep",
    "DataModelsStep",
    "EntryPointsStep",
    "ExternalDependenciesStep",
    "FunctionAnalyticsStep",
    "FunctionContractsStep",
    "FunctionEffectsStep",
    "FunctionHistoryStep",
    "GraphMetricsStep",
    "HistoryTimeseriesStep",
    "HotspotsStep",
    "ProfilesStep",
    "RiskFactorsStep",
    "SemanticRolesStep",
    "SubsystemsStep",
    "TestCoverageEdgesStep",
    "TestProfileStep",
]
