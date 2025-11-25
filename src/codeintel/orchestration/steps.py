"""Pipeline step definitions for ingestion, graphs, analytics, and export."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import duckdb
from coverage import Coverage

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg_metrics import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.function_contracts import compute_function_contracts
from codeintel.analytics.function_effects import compute_function_effects
from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.functions import compute_function_metrics_and_types
from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.analytics.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.history_timeseries import compute_history_timeseries
from codeintel.analytics.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.test_graph_metrics import compute_test_graph_metrics
from codeintel.analytics.test_profiles import build_behavioral_coverage, build_test_profile
from codeintel.analytics.tests_analytics import compute_test_coverage_edges
from codeintel.config.models import (
    BehavioralCoverageConfig,
    CallGraphConfig,
    CFGBuilderConfig,
    ConfigDataFlowConfig,
    CoverageAnalyticsConfig,
    DataModelsConfig,
    DataModelUsageConfig,
    EntryPointsConfig,
    ExternalDependenciesConfig,
    FunctionAnalyticsConfig,
    FunctionAnalyticsOverrides,
    FunctionContractsConfig,
    FunctionEffectsConfig,
    FunctionHistoryConfig,
    GoidBuilderConfig,
    GraphMetricsConfig,
    HistoryTimeseriesConfig,
    HotspotsConfig,
    ImportGraphConfig,
    ProfilesAnalyticsConfig,
    SemanticRolesConfig,
    SubsystemsConfig,
    SymbolUsesConfig,
    TestCoverageConfig,
    TestProfileConfig,
    ToolsConfig,
)
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.function_catalog_service import FunctionCatalogService
from codeintel.graphs.goid_builder import build_goids
from codeintel.graphs.import_graph import build_import_graph
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.graphs.validation import run_graph_validations
from codeintel.ingestion.runner import (
    IngestionContext,
    run_ast_extract,
    run_config_ingest,
    run_coverage_ingest,
    run_cst_extract,
    run_docstrings_ingest,
    run_repo_scan,
    run_scip_ingest,
    run_tests_ingest,
    run_typing_ingest,
)
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.models.rows import CallGraphEdgeRow, CFGBlockRow, CFGEdgeRow, DFGEdgeRow
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)


def _parse_commits(commits_extra: object, commits_env: str) -> tuple[str, ...]:
    """Normalize commit configuration from env vars and pipeline extras."""
    commits_from_env = tuple(commit for commit in commits_env.split(",") if commit)
    if isinstance(commits_extra, str):
        commits_from_extra = tuple(commit for commit in commits_extra.split(",") if commit)
    elif isinstance(commits_extra, Iterable):
        commits_from_extra = tuple(str(commit) for commit in commits_extra)
    else:
        commits_from_extra = ()
    return tuple(commit for commit in (*commits_from_extra, *commits_from_env) if commit)


def _log_step(name: str) -> None:
    """Log step execution at debug level."""
    log.debug("Running pipeline step: %s", name)


@dataclass
class PipelineContext:
    """
    Shared context passed to every pipeline step.

    This matches the repo layout described in your architecture:
      repo_root/
        src/
        Document Output/
        build/
    """

    repo_root: Path
    db_path: Path
    build_dir: Path
    repo: str
    commit: str
    gateway: StorageGateway
    tools: ToolsConfig | None = None
    scan_config: ScanConfig | None = None
    tool_runner: ToolRunner | None = None
    coverage_loader: Callable[[TestCoverageConfig], Coverage | None] | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    function_catalog: FunctionCatalogService | None = None
    extra: dict[str, object] = field(default_factory=dict)
    function_overrides: FunctionAnalyticsOverrides | None = None

    @property
    def document_output_dir(self) -> Path:
        """Document Output directory resolved under repo root."""
        return self.repo_root / "Document Output"


def _ingestion_ctx(ctx: PipelineContext) -> IngestionContext:
    """
    Build an ingestion context from a pipeline context.

    Returns
    -------
    IngestionContext
        Normalized ingestion context for downstream runners.
    """
    return IngestionContext(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        db_path=ctx.db_path,
        build_dir=ctx.build_dir,
        document_output_dir=ctx.document_output_dir,
        gateway=ctx.gateway,
        tools=ctx.tools,
        scan_config=ctx.scan_config,
        tool_runner=ctx.tool_runner,
        scip_runner=ctx.scip_runner,
    )


def _function_catalog(ctx: PipelineContext) -> FunctionCatalogService:
    """
    Return a cached FunctionCatalogService, constructing it on first access.

    Parameters
    ----------
    ctx
        Pipeline context carrying previously constructed catalog if available.

    Returns
    -------
    FunctionCatalogService
        Catalog service for the current repo and commit.
    """
    if ctx.function_catalog is None:
        ctx.function_catalog = FunctionCatalogService.from_db(
            ctx.gateway, repo=ctx.repo, commit=ctx.commit
        )
    return ctx.function_catalog


def _seed_catalog_modules(
    gateway: StorageGateway,
    catalog: FunctionCatalogService | None,
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


class PipelineStep(Protocol):
    """Contract for pipeline steps."""

    name: str
    deps: Sequence[str]

    def run(self, ctx: PipelineContext) -> None:
        """Execute the step using shared context."""


# ---------------------------------------------------------------------------
# Ingestion steps
# ---------------------------------------------------------------------------


@dataclass
class SchemaBootstrapStep:
    """Apply schemas and create views before ingestion."""

    name: str = "schema_bootstrap"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:  # noqa: ARG002
        """No-op here; actual bootstrap is handled in the Prefect task."""
        _log_step(self.name)


@dataclass
class RepoScanStep:
    """Ingest repository modules and repo_map."""

    name: str = "repo_scan"
    deps: Sequence[str] = ("schema_bootstrap",)

    def run(self, ctx: PipelineContext) -> None:
        """Execute repository scan ingestion."""
        _log_step(self.name)
        run_repo_scan(_ingestion_ctx(ctx))


@dataclass
class SCIPIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Register SCIP artifacts and populate SCIP symbols in crosswalk."""
        _log_step(self.name)
        ingest_ctx = _ingestion_ctx(ctx)
        result = run_scip_ingest(ingest_ctx)
        ctx.extra["scip_ingest"] = result
        if result.status != "success":
            log.info("SCIP ingestion %s: %s", result.status, result.reason or "no reason provided")


@dataclass
class CSTStep:
    """Parse CST and persist rows."""

    name: str = "cst_extract"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Extract CST rows into core.cst_nodes."""
        _log_step(self.name)
        run_cst_extract(_ingestion_ctx(ctx))


@dataclass
class AstStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Extract AST rows and metrics into core tables."""
        _log_step(self.name)
        run_ast_extract(_ingestion_ctx(ctx))


@dataclass
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest line-level coverage signals."""
        _log_step(self.name)
        run_coverage_ingest(_ingestion_ctx(ctx))


@dataclass
class TestsIngestStep:
    """Load pytest JSON report into analytics.test_catalog."""

    name: str = "tests_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest pytest test catalog."""
        _log_step(self.name)
        run_tests_ingest(_ingestion_ctx(ctx))


@dataclass
class TypingIngestStep:
    """Collect typedness/static diagnostics."""

    name: str = "typing_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest typing signals from ast + pyright."""
        _log_step(self.name)
        run_typing_ingest(_ingestion_ctx(ctx))


@dataclass
class DocstringsIngestStep:
    """Extract and persist structured docstrings."""

    name: str = "docstrings_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest docstrings for all Python modules."""
        _log_step(self.name)
        run_docstrings_ingest(_ingestion_ctx(ctx))


@dataclass
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        """Ingest configuration files from repo root."""
        _log_step(self.name)
        run_config_ingest(_ingestion_ctx(ctx))


# ---------------------------------------------------------------------------
# Graph steps
# ---------------------------------------------------------------------------


@dataclass
class GoidsStep:
    """Build core.goids and core.goid_crosswalk from AST."""

    name: str = "goids"
    deps: Sequence[str] = ("ast_extract",)

    def run(self, ctx: PipelineContext) -> None:
        """Build GOID registry and crosswalk tables."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = GoidBuilderConfig.from_paths(repo=ctx.repo, commit=ctx.commit, language="python")
        build_goids(gateway, cfg)


@dataclass
class CallGraphStep:
    """Build graph.call_graph_nodes and graph.call_graph_edges."""

    name: str = "callgraph"
    deps: Sequence[str] = ("goids", "repo_scan")

    def run(self, ctx: PipelineContext) -> None:
        """Construct static call graph nodes and edges."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = CallGraphConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            cst_collector=ctx.cst_collector,
            ast_collector=ctx.ast_collector,
        )
        build_call_graph(gateway, cfg, catalog_provider=catalog)


@dataclass
class CFGStep:
    """Build graph.cfg_blocks, graph.cfg_edges, and graph.dfg_edges."""

    name: str = "cfg"
    deps: Sequence[str] = ("function_metrics",)  # falls back to GOIDs if needed

    def run(self, ctx: PipelineContext) -> None:
        """Create minimal CFG/DFG scaffolding."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = CFGBuilderConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            cfg_builder=ctx.cfg_builder,
        )
        build_cfg_and_dfg(gateway, cfg, catalog_provider=catalog)


@dataclass
class ImportGraphStep:
    """Build graph.import_graph_edges from LibCST imports."""

    name: str = "import_graph"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Construct module import graph edges."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ImportGraphConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_import_graph(gateway, cfg)


@dataclass
class SymbolUsesStep:
    """Build graph.symbol_use_edges from index.scip.json."""

    name: str = "symbol_uses"
    deps: Sequence[str] = ("repo_scan", "scip_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Derive symbol definition→use edges from SCIP JSON."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        scip_json = ctx.build_dir / "scip" / "index.scip.json"
        if not scip_json.is_file():
            log.info("Skipping symbol_uses: SCIP JSON missing at %s", scip_json)
            return
        cfg = SymbolUsesConfig.from_paths(
            repo_root=ctx.repo_root,
            scip_json_path=scip_json,
            repo=ctx.repo,
            commit=ctx.commit,
        )
        build_symbol_use_edges(gateway, cfg, catalog_provider=catalog)


@dataclass
class GraphValidationStep:
    """Run integrity validations over graph datasets."""

    name: str = "graph_validation"
    deps: Sequence[str] = ("callgraph", "cfg")

    def run(self, ctx: PipelineContext) -> None:
        """Emit warnings for missing GOIDs, span mismatches, and orphans."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        run_graph_validations(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            catalog_provider=catalog,
            logger=log,
        )


# ---------------------------------------------------------------------------
# Analytics steps
# ---------------------------------------------------------------------------


@dataclass
class HotspotsStep:
    """Build analytics.hotspots from core.ast_metrics plus git churn."""

    name: str = "hotspots"
    deps: Sequence[str] = ("ast_extract",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute file-level hotspot scores."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = HotspotsConfig.from_paths(
            repo_root=ctx.repo_root,
            repo=ctx.repo,
            commit=ctx.commit,
        )
        build_hotspots(gateway, cfg, runner=ctx.tool_runner)


@dataclass
class FunctionHistoryStep:
    """Aggregate per-function git history."""

    name: str = "function_history"
    deps: Sequence[str] = ("function_metrics", "hotspots")

    def run(self, ctx: PipelineContext) -> None:
        """Compute git churn and history for each function GOID."""
        _log_step(self.name)
        cfg = FunctionHistoryConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_history(ctx.gateway.con, cfg, runner=ctx.tool_runner)


@dataclass
class HistoryTimeseriesStep:
    """Aggregate cross-commit analytics.history_timeseries."""

    name: str = "history_timeseries"
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

        def _resolve_db(commit: str) -> duckdb.DuckDBPyConnection:
            target_path = history_db_dir / f"codeintel-{commit}.duckdb"
            if target_path.resolve() == ctx.db_path.resolve():
                return ctx.gateway.con
            if not target_path.is_file():
                message = f"Missing snapshot database for commit {commit}: {target_path}"
                raise FileNotFoundError(message)
            return duckdb.connect(str(target_path), read_only=True)

        cfg = HistoryTimeseriesConfig.from_args(
            repo=ctx.repo,
            repo_root=ctx.repo_root,
            commits=commits,
        )
        compute_history_timeseries(ctx.gateway.con, cfg, _resolve_db, runner=ctx.tool_runner)


@dataclass
class FunctionAnalyticsStep:
    """Build analytics.function_metrics and analytics.function_types."""

    name: str = "function_metrics"
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        """Compute per-function metrics and typedness."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = FunctionAnalyticsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            overrides=ctx.function_overrides,
        )
        summary = compute_function_metrics_and_types(gateway, cfg)
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
    deps: Sequence[str] = ("goids", "callgraph")

    def run(self, ctx: PipelineContext) -> None:
        """Compute function_effects flags and evidence."""
        _log_step(self.name)
        cfg = FunctionEffectsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_effects(ctx.gateway, cfg, catalog_provider=_function_catalog(ctx))


@dataclass
class FunctionContractsStep:
    """Infer pre/postconditions and nullability."""

    name: str = "function_contracts"
    deps: Sequence[str] = ("function_metrics", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Compute inferred contracts for functions."""
        _log_step(self.name)
        cfg = FunctionContractsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_function_contracts(ctx.gateway, cfg, catalog_provider=_function_catalog(ctx))


@dataclass
class DataModelsStep:
    """Extract structured data models from class definitions."""

    name: str = "data_models"
    deps: Sequence[str] = ("ast_extract", "goids", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.data_models."""
        _log_step(self.name)
        cfg = DataModelsConfig.from_paths(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
        compute_data_models(ctx.gateway.con, cfg)


@dataclass
class DataModelUsageStep:
    """Classify per-function data model usage."""

    name: str = "data_model_usage"
    deps: Sequence[str] = ("data_models", "callgraph", "cfg", "function_metrics")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.data_model_usage."""
        _log_step(self.name)
        cfg = DataModelUsageConfig.from_paths(
            repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root
        )
        catalog = _function_catalog(ctx)
        compute_data_model_usage(ctx.gateway, cfg, catalog_provider=catalog)


@dataclass
class ConfigDataFlowStep:
    """Track config key usage at the function level."""

    name: str = "config_data_flow"
    deps: Sequence[str] = ("config_ingest", "callgraph", "function_metrics", "entrypoints")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.config_data_flow."""
        _log_step(self.name)
        cfg = ConfigDataFlowConfig.from_paths(
            repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root
        )
        compute_config_data_flow(ctx.gateway, cfg)


@dataclass
class CoverageAnalyticsStep:
    """Build analytics.coverage_functions from GOIDs and coverage_lines."""

    name: str = "coverage_functions"
    deps: Sequence[str] = ("goids", "coverage_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Aggregate line coverage to function spans."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = CoverageAnalyticsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_coverage_functions(gateway, cfg)


@dataclass
class TestCoverageEdgesStep:
    """Build analytics.test_coverage_edges from coverage contexts."""

    name: str = "test_coverage_edges"
    deps: Sequence[str] = ("coverage_ingest", "tests_ingest", "goids")

    def run(self, ctx: PipelineContext) -> None:
        """Derive test-to-function edges using coverage contexts."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = TestCoverageConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            coverage_loader=ctx.coverage_loader,
        )
        compute_test_coverage_edges(gateway, cfg, catalog_provider=catalog)


@dataclass
class RiskFactorsStep:
    """Aggregate analytics into analytics.goid_risk_factors."""

    name: str = "risk_factors"
    deps: Sequence[str] = (
        "function_metrics",
        "coverage_functions",
        "hotspots",
        "typing_ingest",
        "tests_ingest",
        "test_coverage_edges",
        "config_ingest",  # indirectly for tags/owners via modules
    )

    def run(self, ctx: PipelineContext) -> None:
        """Compute risk factors by joining analytics tables."""
        _log_step(self.name)
        log.info("Computing risk_factors for %s@%s", ctx.repo, ctx.commit)
        gateway = ctx.gateway
        con = gateway.con
        catalog = ctx.function_catalog

        # Clear previous rows for this repo/commit
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
    deps: Sequence[str] = ("callgraph", "import_graph", "symbol_uses", "cfg", "test_coverage_edges")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.graph_metrics_* tables."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = GraphMetricsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        catalog = _function_catalog(ctx)
        compute_graph_metrics(gateway, cfg, catalog_provider=catalog)
        compute_graph_metrics_functions_ext(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_test_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_cfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_dfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_graph_metrics_modules_ext(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_symbol_graph_metrics_modules(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_symbol_graph_metrics_functions(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_config_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_subsystem_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_subsystem_agreement(gateway, repo=ctx.repo, commit=ctx.commit)
        compute_graph_stats(gateway, repo=ctx.repo, commit=ctx.commit)


@dataclass
class SemanticRolesStep:
    """Classify functions and modules into semantic roles."""

    name: str = "semantic_roles"
    deps: Sequence[str] = (
        "function_effects",
        "function_contracts",
        "graph_metrics",
        "function_metrics",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Compute semantic role tables."""
        _log_step(self.name)
        cfg = SemanticRolesConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        compute_semantic_roles(ctx.gateway, cfg, catalog_provider=_function_catalog(ctx))


@dataclass
class SubsystemsStep:
    """Infer subsystems from module coupling and risk signals."""

    name: str = "subsystems"
    deps: Sequence[str] = ("import_graph", "symbol_uses", "risk_factors")

    def run(self, ctx: PipelineContext) -> None:
        """Populate subsystem membership and summaries."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = SubsystemsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        build_subsystems(gateway, cfg)


@dataclass
class TestProfileStep:
    """Build per-test profiles."""

    name: str = "test_profile"
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
        cfg = TestProfileConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        build_test_profile(ctx.gateway, cfg)


@dataclass
class BehavioralCoverageStep:
    """Assign heuristic behavior tags to tests."""

    name: str = "behavioral_coverage"
    deps: Sequence[str] = ("test_profile",)

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.behavioral_coverage."""
        _log_step(self.name)
        enable_llm = bool(
            ctx.extra.get("enable_behavioral_llm")
            or os.getenv("CODEINTEL_BEHAVIORAL_LLM", "").lower() in {"1", "true", "yes"}
        )
        llm_model = ctx.extra.get("behavioral_llm_model")
        llm_runner = ctx.extra.get("behavioral_llm_runner")
        cfg = BehavioralCoverageConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            enable_llm=enable_llm,
            llm_model=llm_model,
        )
        build_behavioral_coverage(ctx.gateway, cfg, llm_runner=llm_runner)  # type: ignore[arg-type]


@dataclass
class EntryPointsStep:
    """Detect HTTP/CLI/job entrypoints and map them to handlers and tests."""

    name: str = "entrypoints"
    deps: Sequence[str] = (
        "subsystems",
        "coverage_functions",
        "test_coverage_edges",
        "goids",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.entrypoints and analytics.entrypoint_tests."""
        _log_step(self.name)
        catalog = _function_catalog(ctx)
        cfg = EntryPointsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            scan_config=ctx.scan_config,
        )
        build_entrypoints(ctx.gateway, cfg, catalog_provider=catalog)


@dataclass
class ExternalDependenciesStep:
    """Identify external dependency usage across functions."""

    name: str = "external_dependencies"
    deps: Sequence[str] = ("goids", "config_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate dependency call edges and aggregated usage."""
        _log_step(self.name)
        catalog = _function_catalog(ctx)
        cfg = ExternalDependenciesConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            scan_config=ctx.scan_config,
        )
        build_external_dependency_calls(ctx.gateway, cfg, catalog_provider=catalog)
        build_external_dependencies(ctx.gateway, cfg)


@dataclass
class ProfilesStep:
    """Build function, file, and module profiles."""

    name: str = "profiles"
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
        catalog = _function_catalog(ctx)
        cfg = ProfilesAnalyticsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        build_function_profile(gateway, cfg, catalog_provider=catalog)
        build_file_profile(gateway, cfg, catalog_provider=catalog)
        build_module_profile(gateway, cfg, catalog_provider=catalog)


# ---------------------------------------------------------------------------
# Export step
# ---------------------------------------------------------------------------


@dataclass
class ExportDocsStep:
    """Export all Parquet + JSONL datasets into Document Output/."""

    name: str = "export_docs"
    deps: Sequence[str] = (
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "coverage_ingest",
        "tests_ingest",
        "typing_ingest",
        "docstrings_ingest",
        "config_ingest",
        "function_metrics",
        "function_effects",
        "function_contracts",
        "data_models",
        "data_model_usage",
        "config_data_flow",
        "coverage_functions",
        "test_coverage_edges",
        "hotspots",
        "function_history",
        "risk_factors",
        "graph_metrics",
        "subsystems",
        "semantic_roles",
        "entrypoints",
        "external_dependencies",
        "test_profile",
        "behavioral_coverage",
        "profiles",
        "history_timeseries",
        "callgraph",
        "cfg",
        "import_graph",
        "symbol_uses",
        "graph_validation",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Create views and export Parquet/JSONL artifacts."""
        _log_step(self.name)
        con = ctx.gateway.con
        create_all_views(con)
        export_all_parquet(ctx.gateway, ctx.document_output_dir)
        export_all_jsonl(ctx.gateway, ctx.document_output_dir)
        log.info("Document Output refreshed at %s", ctx.document_output_dir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIPELINE_STEPS: dict[str, PipelineStep] = {
    # ingestion
    "schema_bootstrap": SchemaBootstrapStep(),
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "cst_extract": CSTStep(),
    "ast_extract": AstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
    "docstrings_ingest": DocstringsIngestStep(),
    "config_ingest": ConfigIngestStep(),
    # graphs
    "goids": GoidsStep(),
    "callgraph": CallGraphStep(),
    "cfg": CFGStep(),
    "import_graph": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
    "graph_validation": GraphValidationStep(),
    # analytics
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
    # export
    "export_docs": ExportDocsStep(),
}
