"""Analytics-focused pipeline steps."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)
from codeintel.analytics.graph_service import build_graph_context
from codeintel.analytics.graphs import (
    compute_config_data_flow,
    compute_config_graph_metrics,
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
    compute_subsystem_agreement,
    compute_subsystem_graph_metrics,
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.tests import (
    build_behavioral_coverage,
    build_test_profile,
    compute_test_coverage_edges,
    compute_test_graph_metrics,
)
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    StepPhase,
    _analytics_context,
    _function_catalog,
    _log_step,
    _resolve_code_profile,
    ensure_graph_runtime,
)
from codeintel.storage.gateway import StorageGateway, build_snapshot_gateway_resolver

log = logging.getLogger(__name__)


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
        gateway = ctx.gateway
        cfg = ctx.config_builder().hotspots()
        build_hotspots(gateway, cfg, runner=ctx.tool_runner)


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
        acx = _analytics_context(ctx)
        compute_function_history(ctx.gateway, cfg, runner=ctx.tool_runner, context=acx)


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
        compute_history_timeseries_gateways(
            ctx.gateway,
            cfg,
            snapshot_resolver,
            runner=ctx.tool_runner,
        )


@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types."""

    name: str = "function_metrics"
    description: str = "Compute per-function metrics, complexity, and type annotations."
    phase: StepPhase = StepPhase.ANALYTICS
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute per-function metrics and typedness."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().function_analytics(
            fail_on_missing_spans=ctx.function_fail_on_missing_spans,
            parser=ctx.function_parser,
        )
        acx = _analytics_context(ctx)
        summary = compute_function_metrics_and_types(
            gateway,
            cfg,
            options=FunctionAnalyticsOptions(context=acx),
        )
        log.info(
            "function_metrics summary rows=%d types=%d validation=%d "
            "parse_failed=%d span_not_found=%d",
            summary["metrics_rows"],
            summary["types_rows"],
            summary["validation_total"],
            summary["validation_parse_failed"],
            summary["validation_span_not_found"],
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
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        compute_function_effects(
            ctx.gateway,
            cfg,
            catalog_provider=acx.catalog,
            context=acx,
            runtime=runtime,
        )


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
        acx = _analytics_context(ctx)
        compute_function_contracts(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
        compute_data_models(ctx.gateway, cfg)


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
        acx = _analytics_context(ctx)
        compute_data_model_usage(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        compute_config_data_flow(ctx.gateway, cfg, context=acx, runtime=runtime)


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
        gateway = ctx.gateway
        cfg = ctx.config_builder().coverage_analytics()
        acx = _analytics_context(ctx)
        compute_coverage_functions(gateway, cfg, context=acx)


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
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = ctx.config_builder().test_coverage(coverage_loader=ctx.coverage_loader)
        compute_test_coverage_edges(gateway, cfg, catalog_provider=catalog)


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
        log.info("Computing risk_factors for %s@%s", ctx.repo, ctx.commit)
        gateway = ctx.gateway
        con = gateway.con
        catalog = ctx.function_catalog

        con.execute(
            "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        )

        use_catalog_modules = _seed_catalog_modules(
            gateway, catalog, repo=ctx.repo, commit=ctx.commit
        )

        risk_sql = """
        INSERT INTO analytics.goid_risk_factors
        SELECT
            fm.function_goid_h128,
            fm.urn,
            fm.repo,
            fm.commit,
            fm.rel_path,
            fm.language,
            fm.kind,
            fm.qualname,
            fm.loc,
            fm.logical_loc,
            fm.cyclomatic_complexity,
            fm.complexity_bucket,
            ft.typedness_bucket,
            ft.typedness_source,
            h.score                       AS hotspot_score,
            CAST(ty.annotation_ratio->>'params' AS DOUBLE) AS file_typed_ratio,
            sd.total_errors               AS static_error_count,
            sd.has_errors                 AS has_static_errors,
            cf.executable_lines,
            cf.covered_lines,
            cf.coverage_ratio,
            cf.tested,
            COALESCE(t_stats.test_count, 0)         AS test_count,
            COALESCE(t_stats.failing_test_count, 0) AS failing_test_count,
            COALESCE(t_stats.last_test_status, 'unknown') AS last_test_status,
            (
                COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                CASE fm.complexity_bucket
                    WHEN 'high' THEN 0.4
                    WHEN 'medium' THEN 0.2
                    ELSE 0.0
                END +
                CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
            ) AS risk_score,
            CASE
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.7 THEN 'high'
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.4 THEN 'medium'
                ELSE 'low'
            END AS risk_level,
            m.tags,
            m.owners,
            NOW() AS created_at
        FROM analytics.function_metrics fm
        LEFT JOIN analytics.function_types ft
            ON ft.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.coverage_functions cf
            ON cf.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.hotspots h
            ON h.rel_path = fm.rel_path
        LEFT JOIN analytics.typedness ty
            ON ty.path = fm.rel_path
        LEFT JOIN analytics.static_diagnostics sd
            ON sd.rel_path = fm.rel_path
        LEFT JOIN (
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id) AS test_count,
                COUNT(
                    DISTINCT CASE WHEN t.status IN ('failed','error') THEN e.test_id END
                ) AS failing_test_count,
                CASE
                    WHEN COUNT(DISTINCT e.test_id) = 0 THEN 'untested'
                    WHEN COUNT(
                        DISTINCT CASE WHEN t.status IN ('failed','error') THEN e.test_id END
                    ) > 0
                        THEN 'some_failing'
                    WHEN COUNT(DISTINCT CASE WHEN t.status = 'passed' THEN e.test_id END) > 0
                        THEN 'all_passing'
                    ELSE 'unknown'
                END AS last_test_status
            FROM analytics.test_coverage_edges e
            LEFT JOIN analytics.test_catalog t
                ON t.test_id = e.test_id
            GROUP BY e.function_goid_h128
        ) AS t_stats
            ON t_stats.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN core.modules m
            ON m.path = fm.rel_path
        WHERE fm.repo = ?
          AND fm.commit = ?;
        """
        if use_catalog_modules:
            risk_sql = risk_sql.replace("core.modules", "temp.catalog_modules")

        con.execute(risk_sql, [ctx.repo, ctx.commit])

        count_row = con.execute(
            "SELECT COUNT(*) FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchone()
        n = int(count_row[0]) if count_row is not None else 0
        log.info("risk_factors populated: %d rows for %s@%s", n, ctx.repo, ctx.commit)


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
        cfg = ctx.config_builder().graph_metrics()
        graph_ctx = build_graph_context(
            cfg,
            now=datetime.now(tz=UTC),
            use_gpu=ctx.graph_backend.use_gpu,
        )
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        compute_graph_metrics(
            gateway,
            cfg,
            catalog_provider=acx.catalog,
            runtime=runtime,
            graph_ctx=graph_ctx,
        )
        compute_graph_metrics_functions_ext(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_test_graph_metrics(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_cfg_metrics(
            gateway, repo=ctx.repo, commit=ctx.commit, context=acx, graph_ctx=graph_ctx
        )
        compute_dfg_metrics(
            gateway, repo=ctx.repo, commit=ctx.commit, context=acx, graph_ctx=graph_ctx
        )
        compute_graph_metrics_modules_ext(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_symbol_graph_metrics_modules(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_symbol_graph_metrics_functions(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_config_graph_metrics(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_subsystem_graph_metrics(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
        )
        compute_subsystem_agreement(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_graph_stats(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            runtime=runtime,
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
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        compute_semantic_roles(
            ctx.gateway,
            cfg,
            catalog_provider=acx.catalog,
            context=acx,
            runtime=runtime,
        )


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
        gateway = ctx.gateway
        cfg = ctx.config_builder().subsystems()
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        build_subsystems(gateway, cfg, context=acx, engine=runtime.engine)


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
        build_test_profile(ctx.gateway, cfg)


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
        build_behavioral_coverage(ctx.gateway, cfg, llm_runner=llm_runner)  # type: ignore[arg-type]


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
        acx = _analytics_context(ctx)
        cfg = ctx.config_builder().entrypoints(scan_profile=_resolve_code_profile(ctx))
        build_entrypoints(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
        acx = _analytics_context(ctx)
        cfg = ctx.config_builder().external_dependencies(
            scan_profile=_resolve_code_profile(ctx),
        )
        build_external_dependency_calls(
            ctx.gateway,
            cfg,
            catalog_provider=acx.catalog,
            context=acx,
        )
        build_external_dependencies(ctx.gateway, cfg)


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
        gateway = ctx.gateway
        acx = _analytics_context(ctx)
        cfg = ctx.config_builder().profiles_analytics()
        build_function_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
        build_file_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
        build_module_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
