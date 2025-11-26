"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import cast

import networkx as nx
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.logging.configuration import setup_logging
from prefect.logging.handlers import PrefectConsoleHandler

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
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
from codeintel.analytics.function_contracts import compute_function_contracts
from codeintel.analytics.function_effects import compute_function_effects
from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.analytics.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import build_graph_context
from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
from codeintel.analytics.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.parsing.validation import FunctionValidationReporter
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.tests import (
    build_behavioral_coverage,
    build_test_profile,
    compute_test_coverage_edges,
    compute_test_graph_metrics,
)
from codeintel.cli.nx_backend import maybe_enable_nx_gpu
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
    SemanticRolesConfig,
    SubsystemsConfig,
    SymbolUsesConfig,
    TestCoverageConfig,
    TestProfileConfig,
    ToolsConfig,
)
from codeintel.core.config import (
    ExecutionConfig,
    ExecutionOptions,
    PathsConfig,
    ScanProfilesConfig,
    SnapshotConfig,
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
from codeintel.ingestion import scip_ingest
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
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.orchestration.steps import (
    PipelineContext,
    ProfilesStep,
    RiskFactorsStep,
    run_pipeline,
)
from codeintel.services.errors import ExportError, log_problem, problem
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from codeintel.storage.schemas import assert_schema_alignment, ensure_schemas_preserve
from codeintel.storage.views import create_all_views


@dataclass(frozen=True)
class ExportArgs:
    """Inputs for the export_docs Prefect flow."""

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    serve_db_path: Path | None = None
    log_db_path: Path | None = None
    tools: ToolsConfig | None = None
    code_profile: ScanProfile | None = None
    config_profile: ScanProfile | None = None
    function_overrides: FunctionAnalyticsOverrides | None = None
    validate_exports: bool = False
    export_schemas: list[str] | None = None
    history_commits: tuple[str, ...] | None = None
    history_db_dir: Path | None = None
    graph_backend: GraphBackendConfig | None = None

    def snapshot_config(self) -> SnapshotConfig:
        """Build a snapshot configuration from the provided arguments.

        Returns
        -------
        SnapshotConfig
            Normalized snapshot descriptor for the flow.
        """
        return SnapshotConfig(
            repo_root=self.repo_root,
            repo_slug=self.repo,
            commit=self.commit,
        )

    def execution_config(self) -> ExecutionConfig:
        """Build the execution config with tool and scan profile defaults applied.

        Returns
        -------
        ExecutionConfig
            Execution settings including tools, profiles, and history options.
        """
        tools_cfg = _tools_from_env(self.tools)
        code_profile = self.code_profile or profile_from_env(default_code_profile(self.repo_root))
        config_profile = self.config_profile or profile_from_env(
            default_config_profile(self.repo_root)
        )
        graph_backend = self.graph_backend or GraphBackendConfig()
        profiles = ScanProfilesConfig(code=code_profile, config=config_profile)
        return ExecutionConfig.for_default_pipeline(
            build_dir=self.build_dir,
            tools=tools_cfg,
            profiles=profiles,
            graph_backend=graph_backend,
            options=ExecutionOptions(
                history_db_dir=self.history_db_dir,
                history_commits=self.history_commits or (),
                function_overrides=(),
            ),
        )

    def storage_config(self) -> StorageConfig:
        """Return an ingest-capable storage configuration.

        Returns
        -------
        StorageConfig
            Gateway configuration for ingest mode.
        """
        return StorageConfig.for_ingest(self.db_path, history_db_path=self.history_db_dir)

    def paths_config(self, execution: ExecutionConfig) -> PathsConfig:
        """Derive build paths for the current snapshot/execution pair.

        Returns
        -------
        PathsConfig
            Derived build and artifact paths bound to the execution config.
        """
        return PathsConfig(
            snapshot=self.snapshot_config(),
            execution=execution,
        )


_GATEWAY_CACHE: dict[
    tuple[str, str, bool, bool, bool, bool, bool],
    StorageGateway,
] = {}
_GATEWAY_STATS: dict[str, int] = {"opens": 0, "hits": 0}
_PREFECT_LOGGING_SETTINGS_PATH = Path(__file__).with_name("prefect_logging.yml")
os.environ.setdefault("PREFECT_LOGGING_SETTINGS_PATH", str(_PREFECT_LOGGING_SETTINGS_PATH))
_GRAPH_BACKEND_STATE: dict[str, GraphBackendConfig] = {"config": GraphBackendConfig()}


def _build_pipeline_context(args: ExportArgs) -> PipelineContext:
    """
    Construct a PipelineContext from export arguments using consolidated configs.

    Returns
    -------
    PipelineContext
        Context ready for pipeline execution.
    """
    snapshot = args.snapshot_config()
    execution = args.execution_config()
    paths = PathsConfig(snapshot=snapshot, execution=execution)
    storage_config = args.storage_config()
    gateway = _get_gateway(storage_config)
    tool_runner = ToolRunner(
        tools_config=execution.tools,
        cache_dir=paths.tool_cache,
    )
    tool_service = ToolService(runner=tool_runner, tools_config=execution.tools)
    extra: dict[str, object] = {}
    if execution.history_commits:
        extra["history_commits"] = execution.history_commits
    elif args.history_commits is not None:
        extra["history_commits"] = args.history_commits
    return PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
        tool_runner=tool_runner,
        tool_service=tool_service,
        tools=execution.tools,
        function_overrides=args.function_overrides,
        extra=extra,
    )


def _gateway_cache_key(config: StorageConfig) -> tuple[str, str, bool, bool, bool, bool, bool]:
    history = str(config.history_db_path.resolve()) if config.history_db_path is not None else ""
    return (
        str(config.db_path.resolve()),
        history,
        config.read_only,
        config.apply_schema,
        config.ensure_views,
        config.validate_schema,
        config.attach_history,
    )


def _get_gateway(config: StorageConfig) -> StorageGateway:
    """
    Return a cached StorageGateway for the flow run.

    Cache key includes all toggles to avoid mixing incompatible configs.

    Returns
    -------
    StorageGateway
        Cached gateway bound to the provided db_path.
    """
    key = _gateway_cache_key(config)
    cached = _GATEWAY_CACHE.get(key)
    if cached is not None:
        _GATEWAY_STATS["hits"] += 1
        return cached
    gateway = open_gateway(config)
    _GATEWAY_STATS["opens"] += 1
    _GATEWAY_CACHE[key] = gateway
    return gateway


def _ingest_config(db_path: Path, *, history_db_path: Path | None = None) -> StorageConfig:
    """Return a write-capable StorageConfig for the primary pipeline database.

    Returns
    -------
    StorageConfig
        Configured for ingest mode with optional history attachment.
    """
    return StorageConfig.for_ingest(db_path, history_db_path=history_db_path)


def _ingest_gateway(db_path: Path, *, history_db_path: Path | None = None) -> StorageGateway:
    """Open or reuse an ingest-mode gateway for the given database path.

    Returns
    -------
    StorageGateway
        Cached or newly opened gateway.
    """
    return _get_gateway(_ingest_config(db_path, history_db_path=history_db_path))


def _close_gateways() -> None:
    """Close and clear any cached gateways."""
    for gateway in _GATEWAY_CACHE.values():
        gateway.close()
    _GATEWAY_CACHE.clear()
    _GATEWAY_STATS["opens"] = 0
    _GATEWAY_STATS["hits"] = 0


def gateway_cache_stats() -> dict[str, int]:
    """
    Return cache statistics for flow gateway reuse.

    Returns
    -------
    dict[str, int]
        Dictionary containing opens, hits, and current cache size.
    """
    return {
        "opens": _GATEWAY_STATS["opens"],
        "hits": _GATEWAY_STATS["hits"],
        "size": len(_GATEWAY_CACHE),
    }


def _graph_backend_config() -> GraphBackendConfig:
    """
    Return the active graph backend configuration for the flow.

    Returns
    -------
    GraphBackendConfig
        Backend preferences currently in effect.
    """
    return _GRAPH_BACKEND_STATE["config"]


def _build_prefect_analytics_context(
    gateway: StorageGateway,
    repo_root: Path,
    repo: str,
    commit: str,
    graph_backend: GraphBackendConfig | None = None,
) -> AnalyticsContext:
    """
    Construct an AnalyticsContext for Prefect tasks.

    Returns
    -------
    AnalyticsContext
        Shared analytics artifacts for the provided repository snapshot.
    """
    backend = graph_backend or _graph_backend_config()
    return build_analytics_context(
        gateway,
        AnalyticsContextConfig(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            use_gpu=backend.use_gpu,
        ),
    )


# ---------------------------------------------------------------------------
# Tasks with light caching/retries for deterministic steps
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)
_PREFECT_LOGGING_CONFIGURED = False


def _configure_prefect_logging() -> None:
    """
    Configure Prefect logging to use a simple stderr handler instead of Rich console.

    Prefect's default console handler can attempt to write to a closed stream during
    teardown of test flows; this replaces console handlers with a standard stream
    handler and leaves other handlers intact.
    """
    global _PREFECT_LOGGING_CONFIGURED  # noqa: PLW0603
    if _PREFECT_LOGGING_CONFIGURED:
        return
    os.environ["PREFECT_LOGGING_SETTINGS_PATH"] = str(_PREFECT_LOGGING_SETTINGS_PATH)
    setup_logging(incremental=False)
    stderr_handler = logging.StreamHandler(sys.__stderr__)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    for logger_name in ("prefect", "prefect.server", "prefect.server.api.server"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [h for h in logger.handlers if not isinstance(h, PrefectConsoleHandler)]
        logger.handlers = []
        logger.addHandler(stderr_handler)
        logger.propagate = False
    root_logger = logging.getLogger()
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, PrefectConsoleHandler)
    ]
    root_logger.addHandler(stderr_handler)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers = [
                h for h in logger.handlers if not isinstance(h, PrefectConsoleHandler)
            ]
            if stderr_handler not in logger.handlers:
                logger.addHandler(stderr_handler)
    _PREFECT_LOGGING_CONFIGURED = True


def _tools_from_env(base: ToolsConfig | None = None) -> ToolsConfig:
    """
    Build a ToolsConfig applying environment overrides when present.

    Returns
    -------
    ToolsConfig
        Merged tool configuration honoring environment variables.
    """
    data = base.model_dump() if base is not None else {}
    env_map = {
        "CODEINTEL_SCIP_PYTHON_BIN": "scip_python_bin",
        "CODEINTEL_SCIP_BIN": "scip_bin",
        "CODEINTEL_PYRIGHT_BIN": "pyright_bin",
        "CODEINTEL_PYREFLY_BIN": "pyrefly_bin",
        "CODEINTEL_RUFF_BIN": "ruff_bin",
        "CODEINTEL_COVERAGE_BIN": "coverage_bin",
        "CODEINTEL_PYTEST_BIN": "pytest_bin",
        "CODEINTEL_GIT_BIN": "git_bin",
        "CODEINTEL_COVERAGE_FILE": "coverage_file",
        "CODEINTEL_PYTEST_REPORT": "pytest_report_path",
    }
    for env_var, field in env_map.items():
        value = os.getenv(env_var)
        if value:
            data[field] = value
    return ToolsConfig.model_validate(data)


def _code_profile_from_env(repo_root: Path, base: ScanProfile | None = None) -> ScanProfile:
    """Build a code ScanProfile honoring environment overrides.

    Returns
    -------
    ScanProfile
        Effective code scan profile after applying overrides.
    """
    return profile_from_env(base or default_code_profile(repo_root))


def _config_profile_from_env(repo_root: Path, base: ScanProfile | None = None) -> ScanProfile:
    """Build a config ScanProfile honoring environment overrides.

    Returns
    -------
    ScanProfile
        Effective config scan profile after applying overrides.
    """
    return profile_from_env(base or default_config_profile(repo_root))


@task(
    name="repo_scan",
    retries=2,
    retry_delay_seconds=2,
    cache_policy=NO_CACHE,
)
def t_repo_scan(ctx: IngestionContext) -> None:
    """Ingest repository modules and repo_map rows."""
    run_repo_scan(ctx)


@task(
    name="scip_ingest",
    retries=2,
    retry_delay_seconds=5,
    cache_policy=NO_CACHE,
)
def t_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Run scip-python ingestion and register SCIP artifacts.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for SCIP ingestion.
    """
    result = run_scip_ingest(ctx)
    if result.status != "success":
        log.info("SCIP ingestion skipped or failed: %s", result.reason or result.status)
    return result


@task(
    name="cst_extract",
    retries=2,
    retry_delay_seconds=2,
    cache_policy=NO_CACHE,
)
def t_cst_extract(ctx: IngestionContext) -> None:
    """Parse CST and persist into cst_nodes."""
    run_cst_extract(ctx)


@task(
    name="ast_extract",
    retries=2,
    retry_delay_seconds=2,
    cache_policy=NO_CACHE,
)
def t_ast_extract(ctx: IngestionContext) -> None:
    """Parse Python stdlib AST and persist into ast tables."""
    run_ast_extract(ctx)


@task(name="coverage_ingest", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines into analytics.coverage_lines."""
    run_coverage_ingest(ctx)


@task(name="tests_ingest", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_tests_ingest(ctx: IngestionContext) -> None:
    """Load pytest test catalog into analytics.test_catalog."""
    run_tests_ingest(ctx)


@task(name="typing_ingest", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_typing_ingest(ctx: IngestionContext) -> None:
    """Collect typedness/static diagnostics."""
    run_typing_ingest(ctx)


@task(
    name="docstrings_ingest",
    retries=1,
    retry_delay_seconds=2,
    cache_policy=NO_CACHE,
)
def t_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract structured docstrings into core.docstrings."""
    run_docstrings_ingest(ctx)


@task(name="config_ingest", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_config_ingest(ctx: IngestionContext) -> None:
    """Flatten config values into analytics.config_values."""
    run_config_ingest(ctx)


@task(name="goids", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_goids(repo: str, commit: str, db_path: Path) -> None:
    """Build GOID registry and crosswalk."""
    gateway = _ingest_gateway(db_path)
    cfg = GoidBuilderConfig.from_paths(repo=repo, commit=commit, language="python")
    build_goids(gateway, cfg)


@task(name="callgraph", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_callgraph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Construct call graph nodes and edges."""
    gateway = _ingest_gateway(db_path)
    cfg = CallGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_call_graph(gateway, cfg)


@task(name="cfg_dfg", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_cfg(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Emit minimal CFG/DFG scaffolding."""
    gateway = _ingest_gateway(db_path)
    cfg = CFGBuilderConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_cfg_and_dfg(gateway, cfg)


@task(name="import_graph", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_import_graph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Build import graph edges via LibCST analysis."""
    gateway = _ingest_gateway(db_path)
    cfg = ImportGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_import_graph(gateway, cfg)


@task(name="symbol_uses", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_symbol_uses(
    cfg: SymbolUsesConfig,
    db_path: Path,
) -> None:
    """Derive symbol use edges from SCIP JSON."""
    gateway = _ingest_gateway(db_path)
    build_symbol_use_edges(gateway, cfg)


@task(name="hotspots", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_hotspots(
    repo_root: Path, repo: str, commit: str, db_path: Path, runner: ToolRunner | None = None
) -> None:
    """Compute file-level hotspot scores."""
    gateway = _ingest_gateway(db_path)
    cfg = HotspotsConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    build_hotspots(gateway, cfg, runner=runner)


@task(name="function_history", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_function_history(
    repo_root: Path, repo: str, commit: str, db_path: Path, runner: ToolRunner | None = None
) -> None:
    """Aggregate per-function git history."""
    gateway = _ingest_gateway(db_path)
    cfg = FunctionHistoryConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_history(gateway, cfg, runner=runner)


@task(name="function_metrics", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_function_metrics(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    overrides: FunctionAnalyticsOverrides | None = None,
) -> None:
    """Compute per-function metrics and types."""
    run_logger = get_run_logger()
    gateway = _ingest_gateway(db_path)
    cfg = FunctionAnalyticsConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        overrides=overrides,
    )
    reporter = FunctionValidationReporter(repo=repo, commit=commit)
    summary = compute_function_metrics_and_types(
        gateway,
        cfg,
        options=FunctionAnalyticsOptions(validation_reporter=reporter),
    )
    run_logger.info(
        "function_metrics summary rows=%d types=%d validation=%d parse_failed=%d span_not_found=%d",
        summary["metrics_rows"],
        summary["types_rows"],
        summary["validation_total"],
        summary["validation_parse_failed"],
        summary["validation_span_not_found"],
    )


@task(name="function_effects", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_function_effects(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Classify side effects and purity."""
    gateway = _ingest_gateway(db_path)
    cfg = FunctionEffectsConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    compute_function_effects(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )


@task(name="function_contracts", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_function_contracts(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Infer contracts and nullability."""
    gateway = _ingest_gateway(db_path)
    cfg = FunctionContractsConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    compute_function_contracts(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )


@task(name="data_models", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_data_models(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Extract structured data models."""
    gateway = _ingest_gateway(db_path)
    cfg = DataModelsConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_data_models(gateway, cfg)


@task(name="data_model_usage", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_data_model_usage(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Classify model usage per function."""
    gateway = _ingest_gateway(db_path)
    cfg = DataModelUsageConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    compute_data_model_usage(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )


@task(name="coverage_functions", retries=1, retry_delay_seconds=2)
def t_coverage_functions(repo: str, commit: str, db_path: Path) -> None:
    """Aggregate coverage lines to function spans."""
    gateway = _ingest_gateway(db_path)
    cfg = CoverageAnalyticsConfig.from_paths(repo=repo, commit=commit)
    compute_coverage_functions(gateway, cfg)


@task(name="test_coverage_edges", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_test_coverage_edges(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Build test-to-function coverage edges."""
    gateway = _ingest_gateway(db_path)
    cfg = TestCoverageConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_test_coverage_edges(gateway, cfg)


@task(name="test_profile", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_test_profile(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Build per-test profiles."""
    gateway = _ingest_gateway(db_path)
    cfg = TestProfileConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_test_profile(gateway, cfg)


@task(name="behavioral_coverage", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_behavioral_coverage(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Assign behavior tags to tests."""
    gateway = _ingest_gateway(db_path)
    cfg = BehavioralCoverageConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_behavioral_coverage(gateway, cfg)


@task(name="graph_metrics", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_graph_metrics(repo: str, commit: str, db_path: Path) -> None:
    """Compute graph metrics for functions and modules."""
    gateway = _ingest_gateway(db_path)
    cfg = GraphMetricsConfig.from_paths(repo=repo, commit=commit)
    graph_backend = _graph_backend_config()
    graph_ctx = build_graph_context(
        cfg,
        now=datetime.now(tz=UTC),
        use_gpu=graph_backend.use_gpu,
    )
    runtime = GraphRuntimeOptions(
        context=None,
        graph_ctx=graph_ctx,
        graph_backend=graph_backend,
    )
    compute_graph_metrics(gateway, cfg, runtime=runtime)
    compute_graph_metrics_functions_ext(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_graph_metrics_modules_ext(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_symbol_graph_metrics_modules(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_symbol_graph_metrics_functions(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_config_graph_metrics(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_subsystem_graph_metrics(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_subsystem_agreement(gateway, repo=repo, commit=commit)
    compute_graph_stats(gateway, repo=repo, commit=commit, runtime=runtime)
    compute_test_graph_metrics(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )
    compute_cfg_metrics(gateway, repo=repo, commit=commit, graph_ctx=graph_ctx)
    compute_dfg_metrics(gateway, repo=repo, commit=commit, graph_ctx=graph_ctx)


@task(name="semantic_roles", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_semantic_roles(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Classify semantic roles for functions and modules."""
    gateway = _ingest_gateway(db_path)
    cfg = SemanticRolesConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    compute_semantic_roles(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )


@task(name="entrypoints", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_entrypoints(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    code_profile: ScanProfile | None,
) -> None:
    """Detect entrypoints across supported frameworks."""
    gateway = _ingest_gateway(db_path)
    cfg = EntryPointsConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        scan_profile=code_profile,
    )
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    build_entrypoints(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )


@task(name="external_dependencies", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_external_dependencies(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    code_profile: ScanProfile | None,
) -> None:
    """Capture external dependency usage."""
    gateway = _ingest_gateway(db_path)
    cfg = ExternalDependenciesConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        scan_profile=code_profile,
    )
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    build_external_dependency_calls(
        gateway,
        cfg,
        catalog_provider=context.catalog,
        context=context,
    )
    build_external_dependencies(gateway, cfg)


@task(name="config_data_flow", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_config_data_flow(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Track config key usage and call chains."""
    gateway = _ingest_gateway(db_path)
    cfg = ConfigDataFlowConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    compute_config_data_flow(gateway, cfg, context=context)


@task(name="risk_factors", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_risk_factors(repo_root: Path, repo: str, commit: str, db_path: Path, build_dir: Path) -> None:
    """Populate analytics.goid_risk_factors from analytics tables."""
    run_logger = get_run_logger()
    try:
        gateway = _ingest_gateway(db_path)
        catalog = FunctionCatalogService.from_db(gateway, repo=repo, commit=commit)
        step = RiskFactorsStep()
        snapshot = SnapshotConfig(repo_root=repo_root, repo_slug=repo, commit=commit)
        profiles = ScanProfilesConfig(
            code=_code_profile_from_env(repo_root),
            config=_config_profile_from_env(repo_root),
        )
        execution = ExecutionConfig.for_default_pipeline(
            build_dir=build_dir,
            tools=_tools_from_env(),
            profiles=profiles,
            graph_backend=_graph_backend_config(),
        )
        paths = PathsConfig(snapshot=snapshot, execution=execution)
        ctx = PipelineContext(
            snapshot=snapshot,
            execution=execution,
            paths=paths,
            gateway=gateway,
            function_catalog=catalog,
        )
        step.run(ctx)
    except Exception as exc:  # pragma: no cover - error path
        pd = problem(
            code="pipeline.task_failed",
            title="risk_factors task failed",
            detail=str(exc),
            extras={"task": "risk_factors", "repo": repo, "commit": commit},
        )
        log_problem(run_logger, pd)
        raise


@task(name="subsystems", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_subsystems(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Infer subsystem clusters and membership."""
    gateway = _ingest_gateway(db_path)
    cfg = SubsystemsConfig.from_paths(repo=repo, commit=commit)
    context = _build_prefect_analytics_context(gateway, repo_root, repo, commit)
    build_subsystems(gateway, cfg, context=context)


@task(name="profiles", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_profiles(repo_root: Path, repo: str, commit: str, db_path: Path, build_dir: Path) -> None:
    """Build function, file, and module profiles."""
    gateway = _ingest_gateway(db_path)
    step = ProfilesStep()
    snapshot = SnapshotConfig(repo_root=repo_root, repo_slug=repo, commit=commit)
    profiles = ScanProfilesConfig(
        code=_code_profile_from_env(repo_root),
        config=_config_profile_from_env(repo_root),
    )
    execution = ExecutionConfig.for_default_pipeline(
        build_dir=build_dir,
        tools=_tools_from_env(),
        profiles=profiles,
        graph_backend=_graph_backend_config(),
    )
    paths = PathsConfig(snapshot=snapshot, execution=execution)
    ctx = PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )
    step.run(ctx)


@dataclass(frozen=True)
class HistoryTimeseriesTaskParams:
    """Arguments for the history_timeseries Prefect task."""

    repo_root: Path
    repo: str
    commits: tuple[str, ...]
    history_db_dir: Path
    db_path: Path
    runner: ToolRunner | None = None


@task(name="history_timeseries", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_history_timeseries(params: HistoryTimeseriesTaskParams) -> None:
    """Aggregate analytics.history_timeseries using multiple commit snapshots."""
    if not params.commits:
        return
    gateway_cfg = StorageConfig(
        db_path=params.db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    gateway = _get_gateway(gateway_cfg)
    params.history_db_dir.mkdir(parents=True, exist_ok=True)

    cfg = HistoryTimeseriesConfig.from_args(
        repo=params.repo,
        repo_root=params.repo_root,
        commits=params.commits,
    )
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=params.history_db_dir,
        repo=params.repo,
        primary_gateway=gateway,
    )
    compute_history_timeseries_gateways(
        gateway,
        cfg,
        snapshot_resolver,
        runner=params.runner,
    )


@task(name="export_docs", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_export_docs(
    db_path: Path,
    document_output_dir: Path,
    *,
    validate_exports: bool = False,
    schemas: list[str] | None = None,
) -> None:
    """
    Create views and export Parquet/JSONL artifacts.

    Raises
    ------
    ExportError
        If export validation fails for any selected dataset.
    """
    run_logger = get_run_logger()
    gateway = _ingest_gateway(db_path)
    con = gateway.con
    try:
        create_all_views(con)
        export_all_parquet(
            gateway,
            document_output_dir,
            validate_exports=validate_exports,
            schemas=schemas,
        )
        export_all_jsonl(
            gateway,
            document_output_dir,
            validate_exports=validate_exports,
            schemas=schemas,
        )
    except ExportError as exc:
        log_problem(run_logger, exc.problem_detail)
        raise
    except Exception as exc:  # pragma: no cover - error path
        pd = problem(
            code="pipeline.task_failed",
            title="export_docs task failed",
            detail=str(exc),
            extras={"task": "export_docs"},
        )
        log_problem(run_logger, pd)
        raise


@task(name="graph_validation", retries=1, retry_delay_seconds=2, cache_policy=NO_CACHE)
def t_graph_validation(repo: str, commit: str, db_path: Path) -> None:
    """Run graph validation warnings."""
    run_logger = get_run_logger()
    try:
        gateway = _ingest_gateway(db_path)
        catalog = FunctionCatalogService.from_db(gateway, repo=repo, commit=commit)
        run_graph_validations(
            gateway,
            repo=repo,
            commit=commit,
            catalog_provider=catalog,
            logger=cast("Logger", run_logger),
        )
    except (
        DuckDBError,
        nx.NetworkXException,
        ValueError,
        RuntimeError,
    ) as exc:  # pragma: no cover - error path
        pd = problem(
            code="pipeline.task_failed",
            title="graph_validation task failed",
            detail=str(exc),
            extras={"task": "graph_validation", "repo": repo, "commit": commit},
        )
        log_problem(run_logger, pd)
        run_logger.warning("Graph validation failed but continuing: %s", exc)


@flow(name="export_docs_flow")
def export_docs_flow(
    args: ExportArgs,
    targets: Iterable[str] | None = None,
) -> None:
    """Run the CodeIntel pipeline within a Prefect flow."""
    _configure_prefect_logging()
    run_logger = get_run_logger()
    graph_backend = args.graph_backend or GraphBackendConfig()
    _GRAPH_BACKEND_STATE["config"] = graph_backend
    maybe_enable_nx_gpu(graph_backend)

    ctx = _build_pipeline_context(args)
    selected = tuple(targets) if targets is not None else None
    try:
        run_logger.info("Starting pipeline for %s@%s", ctx.repo, ctx.commit)
        run_pipeline(ctx, selected_steps=selected)
        run_logger.info("Pipeline complete for %s@%s", ctx.repo, ctx.commit)
    finally:
        _close_gateways()


@task(name="schema_bootstrap", retries=1, retry_delay_seconds=1, cache_policy=NO_CACHE)
def t_schema_bootstrap(db_path: Path) -> None:
    """
    Apply schemas, validate alignment, and create views.

    This is intentionally a distinct task for visibility and future adjustments.
    """
    gateway = _get_gateway(
        StorageConfig(
            db_path=db_path,
            read_only=False,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
    )
    ensure_schemas_preserve(gateway.con)
    create_all_views(gateway.con)
    assert_schema_alignment(gateway.con, strict=True, logger=log)
