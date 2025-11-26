"""Pipeline step definitions for ingestion, graphs, analytics, and export."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from coverage import Coverage

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.context import (
    AnalyticsContext,
    AnalyticsContextConfig,
    build_analytics_context,
)
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
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import GraphContext, build_graph_context
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
    GraphBackendConfig,
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
from codeintel.core.config import ExecutionConfig, PathsConfig, SnapshotConfig
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.engine_factory import build_graph_engine
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.goid_builder import build_goids
from codeintel.graphs.import_graph import build_import_graph
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.graphs.validation import run_graph_validations
from codeintel.ingestion.change_tracker import ChangeTracker
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
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.models.rows import CallGraphEdgeRow, CFGBlockRow, CFGEdgeRow, DFGEdgeRow
from codeintel.server.datasets import validate_dataset_registry
from codeintel.storage.gateway import (
    StorageGateway,
    build_snapshot_gateway_resolver,
)
from codeintel.storage.views import create_all_views

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

    snapshot: SnapshotConfig
    execution: ExecutionConfig
    paths: PathsConfig
    gateway: StorageGateway
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    coverage_loader: Callable[[TestCoverageConfig], Coverage | None] | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    function_catalog: FunctionCatalogProvider | None = None
    extra: dict[str, object] = field(default_factory=dict)
    function_overrides: FunctionAnalyticsOverrides | None = None
    analytics_context: AnalyticsContext | None = None
    change_tracker: ChangeTracker | None = None
    tools: ToolsConfig | None = None
    graph_engine: NxGraphEngine | None = None
    export_datasets: tuple[str, ...] | None = None

    @property
    def document_output_dir(self) -> Path:
        """Document Output directory resolved under repo root."""
        return self.paths.document_output_dir

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot."""
        return self.snapshot.repo_root

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot."""
        return self.snapshot.repo_slug

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot."""
        return self.snapshot.commit

    @property
    def db_path(self) -> Path:
        """DuckDB backing database path."""
        return self.gateway.config.db_path

    @property
    def build_dir(self) -> Path:
        """Build directory resolved from execution config."""
        return self.execution.build_dir

    @property
    def tools_config(self) -> ToolsConfig:
        """Resolve the active tools configuration."""
        return self.tools or self.execution.tools

    @property
    def code_profile(self) -> ScanProfile:
        """Code scan profile for this run."""
        return self.execution.code_profile

    @property
    def config_profile(self) -> ScanProfile:
        """Config scan profile for this run."""
        return self.execution.config_profile

    @property
    def graph_backend(self) -> GraphBackendConfig:
        """Graph backend configuration."""
        return self.execution.graph_backend


def _resolve_code_profile(ctx: PipelineContext) -> ScanProfile:
    """Return the configured code scan profile.

    Returns
    -------
    ScanProfile
        Code scan profile for the current run.
    """
    return ctx.code_profile


def _resolve_config_profile(ctx: PipelineContext) -> ScanProfile:
    """Return the configured config scan profile.

    Returns
    -------
    ScanProfile
        Config scan profile for the current run.
    """
    return ctx.config_profile


def _ingestion_ctx(ctx: PipelineContext) -> IngestionContext:
    """
    Build an ingestion context from a pipeline context.

    Returns
    -------
    IngestionContext
        Normalized ingestion context for downstream runners.
    """
    return IngestionContext(
        snapshot=ctx.snapshot,
        execution=ctx.execution,
        paths=ctx.paths,
        gateway=ctx.gateway,
        tools=ctx.tools_config,
        tool_runner=ctx.tool_runner,
        tool_service=ctx.tool_service,
        scip_runner=ctx.scip_runner,
        artifact_writer=ctx.artifact_writer,
        change_tracker=ctx.change_tracker,
    )


def _function_catalog(ctx: PipelineContext) -> FunctionCatalogProvider:
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
    if ctx.analytics_context is not None:
        ctx.function_catalog = ctx.analytics_context.catalog
        return ctx.analytics_context.catalog
    if ctx.function_catalog is None:
        ctx.function_catalog = FunctionCatalogService.from_db(
            ctx.gateway, repo=ctx.repo, commit=ctx.commit
        )
    return ctx.function_catalog


def _analytics_context(ctx: PipelineContext) -> AnalyticsContext:
    """
    Return a cached AnalyticsContext built for the current snapshot.

    Returns
    -------
    AnalyticsContext
        Shared analytics artifacts (catalog, module map, graphs, ASTs).
    """
    if ctx.analytics_context is None:
        ctx.analytics_context = build_analytics_context(
            ctx.gateway,
            AnalyticsContextConfig(
                repo=ctx.repo,
                commit=ctx.commit,
                repo_root=ctx.repo_root,
                use_gpu=ctx.graph_backend.use_gpu,
            ),
        )
        ctx.function_catalog = ctx.analytics_context.catalog
    return ctx.analytics_context


def _graph_engine(ctx: PipelineContext, acx: AnalyticsContext | None = None) -> NxGraphEngine:
    """
    Construct or reuse a shared graph engine for the pipeline run.

    Parameters
    ----------
    ctx :
        Pipeline context carrying gateway, backend, and snapshot metadata.
    acx :
        Optional analytics context to seed cache data.

    Returns
    -------
    NxGraphEngine
        Cached engine keyed to the pipeline snapshot.
    """
    context = acx or ctx.analytics_context
    if ctx.graph_engine is None:
        ctx.graph_engine = build_graph_engine(
            ctx.gateway,
            (ctx.repo, ctx.commit),
            graph_backend=ctx.graph_backend,
            context=context,
        )
    return ctx.graph_engine


def _graph_runtime(
    ctx: PipelineContext,
    *,
    graph_ctx: GraphContext | None = None,
    acx: AnalyticsContext | None = None,
) -> GraphRuntimeOptions:
    """
    Build a runtime bundle for graph consumers using the shared engine.

    Parameters
    ----------
    ctx :
        Pipeline context for the current run.
    graph_ctx :
        Optional graph context describing edge weights and limits.
    acx :
        Optional analytics context used for cache seeding.

    Returns
    -------
    GraphRuntimeOptions
        Runtime configured to reuse the shared engine and backend.
    """
    context = acx or _analytics_context(ctx)
    engine = _graph_engine(ctx, context)
    return GraphRuntimeOptions(
        context=context,
        graph_ctx=graph_ctx,
        graph_backend=ctx.graph_backend,
        engine=engine,
    )


def ensure_graph_engine(ctx: PipelineContext, acx: AnalyticsContext | None = None) -> NxGraphEngine:
    """
    Public helper to retrieve the shared graph engine for the pipeline snapshot.

    Returns
    -------
    NxGraphEngine
        Shared engine keyed to the pipeline's repo and commit.
    """
    return _graph_engine(ctx, acx)


def ensure_graph_runtime(
    ctx: PipelineContext,
    *,
    graph_ctx: GraphContext | None = None,
    acx: AnalyticsContext | None = None,
) -> GraphRuntimeOptions:
    """
    Public helper to construct a shared graph runtime for the pipeline snapshot.

    Returns
    -------
    GraphRuntimeOptions
        Runtime bound to the shared engine and backend preferences.
    """
    return _graph_runtime(ctx, graph_ctx=graph_ctx, acx=acx)


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
        tracker = run_repo_scan(_ingestion_ctx(ctx))
        ctx.change_tracker = tracker


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
        acx = _analytics_context(ctx)
        compute_function_history(ctx.gateway, cfg, runner=ctx.tool_runner, context=acx)


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

        cfg = HistoryTimeseriesConfig.from_args(
            repo=ctx.repo,
            repo_root=ctx.repo_root,
            commits=commits,
        )
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
    deps: Sequence[str] = ("goids", "callgraph")

    def run(self, ctx: PipelineContext) -> None:
        """Compute function_effects flags and evidence."""
        _log_step(self.name)
        cfg = FunctionEffectsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        acx = _analytics_context(ctx)
        runtime = _graph_runtime(ctx, acx=acx)
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
    deps: Sequence[str] = ("function_metrics", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Compute inferred contracts for functions."""
        _log_step(self.name)
        cfg = FunctionContractsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
        )
        acx = _analytics_context(ctx)
        compute_function_contracts(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


@dataclass
class DataModelsStep:
    """Extract structured data models from class definitions."""

    name: str = "data_models"
    deps: Sequence[str] = ("ast_extract", "goids", "docstrings_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate analytics.data_models."""
        _log_step(self.name)
        cfg = DataModelsConfig.from_paths(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)
        compute_data_models(ctx.gateway, cfg)


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
        acx = _analytics_context(ctx)
        compute_data_model_usage(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
        acx = _analytics_context(ctx)
        runtime = ensure_graph_runtime(ctx, acx=acx)
        compute_config_data_flow(ctx.gateway, cfg, context=acx, runtime=runtime)


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
        acx = _analytics_context(ctx)
        compute_coverage_functions(gateway, cfg, context=acx)


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
        graph_ctx = build_graph_context(
            cfg,
            now=datetime.now(tz=UTC),
            use_gpu=ctx.graph_backend.use_gpu,
        )
        acx = _analytics_context(ctx)
        runtime = _graph_runtime(ctx, graph_ctx=graph_ctx, acx=acx)
        compute_graph_metrics(
            gateway,
            cfg,
            catalog_provider=acx.catalog,
            runtime=runtime,
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
    deps: Sequence[str] = ("import_graph", "symbol_uses", "risk_factors")

    def run(self, ctx: PipelineContext) -> None:
        """Populate subsystem membership and summaries."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = SubsystemsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        acx = _analytics_context(ctx)
        engine = _graph_engine(ctx, acx)
        build_subsystems(gateway, cfg, context=acx, engine=engine)


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
        llm_model_raw = ctx.extra.get("behavioral_llm_model")
        llm_model = llm_model_raw if isinstance(llm_model_raw, str) else None
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
        acx = _analytics_context(ctx)
        cfg = EntryPointsConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            scan_profile=_resolve_code_profile(ctx),
        )
        build_entrypoints(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


@dataclass
class ExternalDependenciesStep:
    """Identify external dependency usage across functions."""

    name: str = "external_dependencies"
    deps: Sequence[str] = ("goids", "config_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Populate dependency call edges and aggregated usage."""
        _log_step(self.name)
        acx = _analytics_context(ctx)
        cfg = ExternalDependenciesConfig.from_paths(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
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
        cfg = ProfilesAnalyticsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        build_function_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
        build_file_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)
        build_module_profile(gateway, cfg, catalog_provider=acx.catalog, context=acx)


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
        validate_dataset_registry(ctx.gateway)
        datasets = list(ctx.export_datasets) if ctx.export_datasets is not None else None
        export_all_parquet(
            ctx.gateway,
            ctx.document_output_dir,
            datasets=datasets,
        )
        export_all_jsonl(
            ctx.gateway,
            ctx.document_output_dir,
            datasets=datasets,
        )
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

PIPELINE_STEPS_BY_NAME: dict[str, PipelineStep] = PIPELINE_STEPS
PIPELINE_DEPS: dict[str, tuple[str, ...]] = {
    name: tuple(step.deps) for name, step in PIPELINE_STEPS.items()
}
PIPELINE_SEQUENCE: tuple[str, ...] = tuple(PIPELINE_STEPS.keys())


def _topological_order(step_names: Sequence[str]) -> list[str]:
    """Return a topological ordering of the requested pipeline steps.

    Returns
    -------
    list[str]
        Steps ordered to respect declared dependencies.

    Raises
    ------
    RuntimeError
        If a dependency cycle is detected.
    """
    deps = {name: set(PIPELINE_DEPS.get(name, ())) for name in step_names}
    remaining = set(step_names)
    ordered: list[str] = []
    no_deps = [name for name in step_names if not deps[name]]

    while no_deps:
        name = no_deps.pop()
        ordered.append(name)
        remaining.discard(name)
        for other in list(remaining):
            deps[other].discard(name)
            if not deps[other]:
                no_deps.append(other)

    if remaining:
        message = f"Circular dependencies detected: {sorted(remaining)}"
        raise RuntimeError(message)
    return ordered


def run_pipeline(ctx: PipelineContext, *, selected_steps: Sequence[str] | None = None) -> None:
    """
    Execute pipeline steps in topological order using the shared context.

    Parameters
    ----------
    ctx
        PipelineContext containing configs and runtime services.
    selected_steps
        Optional subset of steps to execute; dependencies are included automatically.

    Raises
    ------
    KeyError
        If a requested step name is not registered.
    RuntimeError
        If dependency ordering fails due to a cycle.
    """
    step_names = tuple(selected_steps) if selected_steps is not None else PIPELINE_SEQUENCE

    def _expand_with_deps(name: str, acc: set[str]) -> None:
        if name in acc:
            return
        for dep in PIPELINE_DEPS.get(name, ()):
            _expand_with_deps(dep, acc)
        acc.add(name)

    expanded: set[str] = set()
    for name in step_names:
        if name not in PIPELINE_STEPS_BY_NAME:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)
        _expand_with_deps(name, expanded)

    ordered_names = [name for name in PIPELINE_SEQUENCE if name in expanded]
    try:
        ordered = _topological_order(tuple(ordered_names))
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(str(exc)) from exc
    for name in ordered:
        step = PIPELINE_STEPS_BY_NAME[name]
        step.run(ctx)
