"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Logger, LoggerAdapter
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
    PIPELINE_STEPS,
    PipelineContext,
    ProfilesStep,
    RiskFactorsStep,
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


@dataclass(frozen=True)
class SnapshotArgs:
    """Snapshot context for logging pipeline database state."""

    db_path: Path
    repo: str
    commit: str
    log_db_path: Path
    run_id: str


@dataclass(frozen=True)
class SnapshotRecord:
    """Snapshot result details for persistence."""

    stage: str
    status: str
    counts: dict[str, int]
    error: str | None = None


_GATEWAY_CACHE: dict[
    tuple[str, str, bool, bool, bool, bool, bool],
    StorageGateway,
] = {}
_GATEWAY_STATS: dict[str, int] = {"opens": 0, "hits": 0}
_PREFECT_LOGGING_SETTINGS_PATH = Path(__file__).with_name("prefect_logging.yml")
os.environ.setdefault("PREFECT_LOGGING_SETTINGS_PATH", str(_PREFECT_LOGGING_SETTINGS_PATH))
_GRAPH_BACKEND_STATE: dict[str, GraphBackendConfig] = {"config": GraphBackendConfig()}


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
    """Return a write-capable StorageConfig for the primary pipeline database."""
    return StorageConfig.for_ingest(db_path, history_db_path=history_db_path)


def _ingest_gateway(db_path: Path, *, history_db_path: Path | None = None) -> StorageGateway:
    """Open or reuse an ingest-mode gateway for the given database path."""
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


def _log_task(name: str, logger: Logger | LoggerAdapter) -> float:
    """
    Emit a lightweight breadcrumb before running a task.

    Returns
    -------
    float
        Timestamp at start of the task for duration tracking.
    """
    logger.info("starting task: %s", name)
    return time.perf_counter()


def _log_task_done(name: str, start_ts: float, logger: Logger | LoggerAdapter) -> None:
    """Emit completion with duration for a task."""
    duration = time.perf_counter() - start_ts
    logger.info("finished task: %s (%.2fs)", name, duration)


def _ensure_db_schema(db_path: Path) -> None:
    """Apply schemas and validate alignment for the target database."""
    gateway = _ingest_gateway(db_path)
    ensure_schemas_preserve(gateway.con)
    assert_schema_alignment(gateway.con, strict=True)


def _run_task(
    name: str,
    fn: Callable[..., object],
    logger: Logger | LoggerAdapter,
    *args: object,
    snapshot: SnapshotArgs | None = None,
    **kwargs: object,
) -> None:
    """Run a Prefect task with start/finish logging."""
    if snapshot is not None:
        _snapshot_db_state(snapshot, stage=f"{name}:before", logger=cast("Logger", logger))
    start = _log_task(name, logger)
    fn(*args, **kwargs)
    _log_task_done(name, start, logger)
    if snapshot is not None:
        _snapshot_db_state(snapshot, stage=f"{name}:after", logger=cast("Logger", logger))


def _resolve_output_dir(repo_root: Path) -> Path:
    """
    Determine the document output directory, overridable via CODEINTEL_OUTPUT_DIR.

    Returns
    -------
    Path
        Target directory for exported artifacts.
    """
    env_dir = os.getenv("CODEINTEL_OUTPUT_DIR")
    folder = env_dir if env_dir else "document_output"
    return repo_root / folder


def _resolve_log_db_path(path_arg: Path | None, build_dir: Path) -> Path:
    """Choose the log database path, defaulting to build/db/codeintel_logs.duckdb.

    Returns
    -------
    Path
        Resolved log database path.
    """
    return (
        path_arg.resolve() if path_arg is not None else (build_dir / "db" / "codeintel_logs.duckdb")
    )


def _snapshot_db_state(
    snapshot: SnapshotArgs,
    *,
    stage: str,
    logger: Logger,
) -> None:
    """Log row counts for key tables to aid debugging between tasks."""
    counts: dict[str, int] = {}
    status = "ok"
    error: str | None = None
    gateway: StorageGateway | None = None
    if not snapshot.db_path.exists():
        status = "missing"
        error = "database not found"
    else:
        try:
            # Match the configuration of the main pipeline gateway (read/write)
            # to avoid DuckDB config mismatch errors, but run read-only queries.
            gateway = _ingest_gateway(snapshot.db_path)
        except DuckDBError as exc:
            status = "error"
            error = f"open failed: {exc}"
    if gateway is not None:
        con = gateway.con
        queries = {
            "core.modules": "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
            "core.goids": "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ?",
            "core.goids.module": (
                "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'module'"
            ),
            "core.goids.class": (
                "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'class'"
            ),
            "core.goids.func": (
                "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind IN "
                "('function','method')"
            ),
            "graph.call_graph_nodes": "SELECT COUNT(*) FROM graph.call_graph_nodes",
            "graph.call_graph_edges": "SELECT COUNT(*) FROM graph.call_graph_edges",
            "analytics.graph_validation": (
                "SELECT COUNT(*) FROM analytics.graph_validation WHERE repo = ? AND commit = ?"
            ),
            "analytics.cfg_function_metrics_ext": (
                "SELECT COUNT(*) FROM analytics.cfg_function_metrics_ext "
                "WHERE repo = ? AND commit = ?"
            ),
            "analytics.dfg_function_metrics_ext": (
                "SELECT COUNT(*) FROM analytics.dfg_function_metrics_ext "
                "WHERE repo = ? AND commit = ?"
            ),
        }
        params = [snapshot.repo, snapshot.commit]
        for label, sql in queries.items():
            try:
                row = con.execute(sql, params if "?" in sql else []).fetchone()
                counts[label] = int(row[0]) if row and row[0] is not None else 0
            except DuckDBError:
                counts[label] = -1
    message = (
        f"[snapshot {stage}] status={status} repo={snapshot.repo} commit={snapshot.commit} "
        + " ".join(f"{k}={v}" for k, v in counts.items())
    )
    if error is not None:
        message = f"{message} error={error}"
    logger.info(message)
    _append_log(message)
    record = SnapshotRecord(stage=stage, status=status, counts=counts, error=error)
    _write_snapshot_log(snapshot=snapshot, record=record, logger=logger)


def _append_log(message: str) -> None:
    """Append a timestamped line to build/logs/pipeline.log for offline inspection."""
    log_path = Path("build/logs/pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def _write_snapshot_log(
    *,
    snapshot: SnapshotArgs,
    record: SnapshotRecord,
    logger: Logger,
) -> None:
    """Persist snapshot metadata into the dedicated logging database."""
    log_db_path = snapshot.log_db_path
    log_db_path.parent.mkdir(parents=True, exist_ok=True)
    config = StorageConfig(db_path=log_db_path, read_only=False, validate_schema=False)
    try:
        gateway = _get_gateway(config)
    except DuckDBError as exc:
        logger.warning("Snapshot log skipped: could not open log DB at %s (%s)", log_db_path, exc)
        return
    try:
        con = gateway.con
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
        con.execute(
            "CREATE TABLE IF NOT EXISTS analytics.snapshot_logs ("
            "run_id TEXT,"
            "repo TEXT,"
            "commit TEXT,"
            "stage TEXT,"
            "status TEXT,"
            "counts JSON,"
            "error TEXT,"
            "created_at TIMESTAMPTZ DEFAULT current_timestamp"
            ")"
        )
        con.execute(
            "INSERT INTO analytics.snapshot_logs "
            "(run_id, repo, commit, stage, status, counts, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.run_id,
                snapshot.repo,
                snapshot.commit,
                record.stage,
                record.status,
                json.dumps(record.counts),
                record.error,
            ),
        )
    except DuckDBError as exc:
        logger.warning("Snapshot log write failed: %s", exc)


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
    """Build a code ScanProfile honoring environment overrides."""
    return profile_from_env(base or default_code_profile(repo_root))


def _config_profile_from_env(repo_root: Path, base: ScanProfile | None = None) -> ScanProfile:
    """Build a config ScanProfile honoring environment overrides."""
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
        ctx = PipelineContext(
            repo_root=repo_root,
            db_path=db_path,
            build_dir=build_dir,
            repo=repo,
            commit=commit,
            gateway=gateway,
            code_profile=_code_profile_from_env(repo_root),
            config_profile=_config_profile_from_env(repo_root),
            function_catalog=catalog,
            graph_backend=_graph_backend_config(),
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
    ctx = PipelineContext(
        repo_root=repo_root,
        db_path=db_path,
        build_dir=build_dir,
        repo=repo,
        commit=commit,
        gateway=gateway,
        graph_backend=_graph_backend_config(),
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


def _preflight_checks(
    ctx: IngestionContext,
    *,
    skip_scip: bool = False,
    logger: Logger | LoggerAdapter | None = None,
) -> None:
    """
    Validate environment prerequisites before running the flow.

    Raises
    ------
    RuntimeError
        If required directories or binaries are missing.
    """
    errors: list[str] = []
    repo_root = ctx.repo_root
    build_dir = ctx.build_dir
    db_path = ctx.db_path
    tools = ctx.tools or ToolsConfig(
        scip_python_bin="scip-python",
        scip_bin="scip",
        pyright_bin="pyright",
    )

    if not repo_root.is_dir():
        errors.append(f"repo_root not found: {repo_root}")
    git_dir = repo_root / ".git"
    if not skip_scip:
        scip_python_bin = tools.scip_python_bin
        scip_bin = tools.scip_bin
        if not git_dir.is_dir():
            errors.append("SCIP ingest requires a git repository (.git missing)")
        if shutil.which(scip_python_bin) is None:
            errors.append(f"scip-python binary not found: {scip_python_bin}")
        if shutil.which(scip_bin) is None:
            errors.append(f"scip binary not found: {scip_bin}")

    document_output = _resolve_output_dir(repo_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    document_output.mkdir(parents=True, exist_ok=True)

    if errors:
        details = "; ".join(errors)
        message = f"Prefect preflight failed: {details}"
        raise RuntimeError(message)
    (logger or log).info("Prefect preflight succeeded for %s", repo_root)


def _resolve_validation_settings(args: ExportArgs) -> tuple[bool, list[str] | None]:
    """
    Resolve export validation settings from args and environment overrides.

    Returns
    -------
    tuple[bool, list[str] | None]
        Flag indicating whether to validate exports and optional schema overrides.
    """
    env_value = os.getenv("CODEINTEL_VALIDATE_EXPORTS", "")
    env_enabled = env_value.lower() in {"1", "true", "yes", "on"}
    env_schemas = os.getenv("CODEINTEL_VALIDATE_SCHEMAS")

    schemas = args.export_schemas
    if env_schemas:
        schemas = [schema.strip() for schema in env_schemas.split(",") if schema.strip()]
    return args.validate_exports or env_enabled, schemas


def _resolve_history_commits(args: ExportArgs) -> tuple[str, ...]:
    """
    Determine history commits to include for timeseries aggregation.

    Prefers explicit args, then CODEINTEL_HISTORY_COMMITS env var, and finally
    falls back to the current commit only.

    Returns
    -------
    tuple[str, ...]
        Commit identifiers ordered with the current commit first.
    """
    if args.history_commits:
        commits = list(args.history_commits)
    else:
        env_value = os.getenv("CODEINTEL_HISTORY_COMMITS")
        commits = [c for c in (env_value or "").split(",") if c]
    if not commits:
        commits = [args.commit]
    if args.commit not in commits:
        commits.insert(0, args.commit)
    else:
        commits = [args.commit] + [c for c in commits if c != args.commit]
    return tuple(commits)


@flow(name="export_docs_flow")
def export_docs_flow(
    args: ExportArgs,
    targets: Iterable[str] | None = None,
) -> None:
    """
    Run the CodeIntel pipeline using Prefect orchestration.

    Parameters
    ----------
    args :
        Repository identity and path configuration.
    targets :
        Optional subset of steps to run (in declared order).
    """
    _configure_prefect_logging()
    run_logger = get_run_logger()
    graph_backend = args.graph_backend or GraphBackendConfig()
    _GRAPH_BACKEND_STATE["config"] = graph_backend
    maybe_enable_nx_gpu(graph_backend)
    build_dir = args.build_dir.resolve()
    serve_db_path = args.serve_db_path.resolve() if args.serve_db_path is not None else None
    log_db_path = _resolve_log_db_path(args.log_db_path, build_dir)
    skip_scip = os.getenv("CODEINTEL_SKIP_SCIP", "false").lower() == "true"
    validation_settings = _resolve_validation_settings(args)
    history_commits = _resolve_history_commits(args)
    history_db_dir = (args.history_db_dir or build_dir / "db").resolve()
    tools_cfg = _tools_from_env(args.tools)
    tool_runner = ToolRunner(
        tools_config=tools_cfg,
        cache_dir=build_dir / ".tool_cache",
    )
    tool_service = ToolService(tool_runner, tools_cfg)
    code_profile = _code_profile_from_env(args.repo_root.resolve(), args.code_profile)
    config_profile = _config_profile_from_env(args.repo_root.resolve(), args.config_profile)

    ctx = IngestionContext(
        repo_root=args.repo_root.resolve(),
        repo=args.repo,
        commit=args.commit,
        db_path=args.db_path.resolve(),
        build_dir=build_dir,
        document_output_dir=_resolve_output_dir(args.repo_root.resolve()),
        gateway=_get_gateway(
            StorageConfig(
                db_path=args.db_path.resolve(),
                read_only=False,
                apply_schema=False,  # t_schema_bootstrap handles schema setup non-destructively
                ensure_views=False,  # t_schema_bootstrap handles view creation
                validate_schema=False,  # t_schema_bootstrap handles validation
            )
        ),
        tools=tools_cfg,
        code_profile=code_profile,
        config_profile=config_profile,
        tool_runner=tool_runner,
        tool_service=tool_service,
    )
    db_path = ctx.db_path
    _preflight_checks(ctx, skip_scip=skip_scip, logger=run_logger)
    scip_state: dict[str, scip_ingest.ScipIngestResult | None] = {"result": None}
    snapshot_args = SnapshotArgs(
        db_path=db_path,
        repo=ctx.repo,
        commit=ctx.commit,
        log_db_path=log_db_path,
        run_id=f"{args.repo}:{args.commit}:{int(time.time() * 1000)}",
    )

    step_handlers = [
        (
            "schema_bootstrap",
            lambda: _run_task(
                "schema_bootstrap",
                t_schema_bootstrap,
                run_logger,
                db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "repo_scan",
            lambda: _run_task(
                "repo_scan",
                t_repo_scan,
                run_logger,
                ctx,
                snapshot=snapshot_args,
            ),
        ),
        (
            "scip_ingest",
            lambda: _run_scip_ingest(
                logger=run_logger,
                skip_scip=skip_scip,
                scip_state=scip_state,
                ctx=ctx,
            ),
        ),
        ("cst_extract", lambda: _run_task("cst_extract", t_cst_extract, run_logger, ctx)),
        (
            "ast_extract",
            lambda: _run_task(
                "ast_extract",
                t_ast_extract,
                run_logger,
                ctx,
                snapshot=snapshot_args,
            ),
        ),
        (
            "coverage_ingest",
            lambda: _run_task("coverage_ingest", t_coverage_ingest, run_logger, ctx),
        ),
        ("tests_ingest", lambda: _run_task("tests_ingest", t_tests_ingest, run_logger, ctx)),
        (
            "typing_ingest",
            lambda: _run_task(
                "typing_ingest",
                t_typing_ingest,
                run_logger,
                ctx,
                snapshot=snapshot_args,
            ),
        ),
        (
            "docstrings_ingest",
            lambda: _run_task("docstrings_ingest", t_docstrings_ingest, run_logger, ctx),
        ),
        ("config_ingest", lambda: _run_task("config_ingest", t_config_ingest, run_logger, ctx)),
        (
            "goids",
            lambda: _run_task("goids", t_goids, run_logger, ctx.repo, ctx.commit, ctx.db_path),
        ),
        (
            "callgraph",
            lambda: _run_task(
                "callgraph",
                t_callgraph,
                run_logger,
                ctx.repo,
                ctx.commit,
                ctx.repo_root,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "cfg",
            lambda: _run_task(
                "cfg_dfg", t_cfg, run_logger, ctx.repo, ctx.commit, ctx.repo_root, ctx.db_path
            ),
        ),
        (
            "import_graph",
            lambda: _run_task(
                "import_graph",
                t_import_graph,
                run_logger,
                ctx.repo,
                ctx.commit,
                ctx.repo_root,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "symbol_uses",
            lambda: _run_symbol_uses(
                logger=run_logger,
                scip_state=scip_state,
                ctx=ctx,
            ),
        ),
        (
            "graph_validation",
            lambda: _run_task(
                "graph_validation",
                t_graph_validation,
                run_logger,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "hotspots",
            lambda: _run_task(
                "hotspots",
                t_hotspots,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.tool_runner,
            ),
        ),
        (
            "function_history",
            lambda: _run_task(
                "function_history",
                t_function_history,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.tool_runner,
            ),
        ),
        (
            "function_metrics",
            lambda: _run_task(
                "function_metrics",
                t_function_metrics,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                args.function_overrides,
                snapshot=snapshot_args,
            ),
        ),
        (
            "function_effects",
            lambda: _run_task(
                "function_effects",
                t_function_effects,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "function_contracts",
            lambda: _run_task(
                "function_contracts",
                t_function_contracts,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "data_models",
            lambda: _run_task(
                "data_models",
                t_data_models,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "data_model_usage",
            lambda: _run_task(
                "data_model_usage",
                t_data_model_usage,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "coverage_functions",
            lambda: _run_task(
                "coverage_functions",
                t_coverage_functions,
                run_logger,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
            ),
        ),
        (
            "test_coverage_edges",
            lambda: _run_task(
                "test_coverage_edges",
                t_test_coverage_edges,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "graph_metrics",
            lambda: _run_task(
                "graph_metrics",
                t_graph_metrics,
                run_logger,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "semantic_roles",
            lambda: _run_task(
                "semantic_roles",
                t_semantic_roles,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "entrypoints",
            lambda: _run_task(
                "entrypoints",
                t_entrypoints,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.code_profile,
                snapshot=snapshot_args,
            ),
        ),
        (
            "external_dependencies",
            lambda: _run_task(
                "external_dependencies",
                t_external_dependencies,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.code_profile,
                snapshot=snapshot_args,
            ),
        ),
        (
            "config_data_flow",
            lambda: _run_task(
                "config_data_flow",
                t_config_data_flow,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "risk_factors",
            lambda: _run_task(
                "risk_factors",
                t_risk_factors,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.build_dir,
            ),
        ),
        (
            "subsystems",
            lambda: _run_task(
                "subsystems",
                t_subsystems,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                snapshot=snapshot_args,
            ),
        ),
        (
            "test_profile",
            lambda: _run_task(
                "test_profile",
                t_test_profile,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
            ),
        ),
        (
            "behavioral_coverage",
            lambda: _run_task(
                "behavioral_coverage",
                t_behavioral_coverage,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
            ),
        ),
        (
            "profiles",
            lambda: _run_task(
                "profiles",
                t_profiles,
                run_logger,
                ctx.repo_root,
                ctx.repo,
                ctx.commit,
                ctx.db_path,
                ctx.build_dir,
                snapshot=snapshot_args,
            ),
        ),
        (
            "history_timeseries",
            lambda: _run_task(
                "history_timeseries",
                t_history_timeseries,
                run_logger,
                HistoryTimeseriesTaskParams(
                    repo_root=ctx.repo_root,
                    repo=ctx.repo,
                    commits=history_commits,
                    history_db_dir=history_db_dir,
                    db_path=ctx.db_path,
                    runner=ctx.tool_runner,
                ),
                snapshot=snapshot_args,
            ),
        ),
        (
            "export_docs",
            lambda: _run_task(
                "export_docs",
                t_export_docs,
                run_logger,
                ctx.db_path,
                ctx.document_output_dir,
                validate_exports=validation_settings[0],
                schemas=validation_settings[1],
            ),
        ),
    ]

    _execute_step_handlers(
        step_handlers=step_handlers,
        targets=targets,
        run_logger=run_logger,
        db_path=db_path,
        serve_db_path=serve_db_path,
    )


def _copy_database(source: Path, target: Path, logger: Logger) -> None:
    """
    Copy a staging database to the serving path after a successful flow run.

    Parameters
    ----------
    source
        Source DuckDB path (staging).
    target
        Destination DuckDB path (serving).
    logger
        Logger for status messages.
    """
    if source.resolve() == target.resolve():
        logger.info("Serving DB path matches staging; skipping copy.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(target)
    logger.info("Copied staging DB %s -> %s", source, target)


def _execute_step_handlers(
    *,
    step_handlers: list[tuple[str, Callable[[], None]]],
    targets: Iterable[str] | None,
    run_logger: Logger | LoggerAdapter,
    db_path: Path,
    serve_db_path: Path | None,
) -> None:
    """Execute step handlers in topological order and handle DB promotion."""
    handlers = dict(step_handlers)
    success = False
    try:
        for name in _toposort_targets(targets):
            handler = handlers.get(name)
            if handler is None:
                run_logger.warning("No handler registered for step %s; skipping.", name)
                continue
            handler()
        success = True
    finally:
        _close_gateways()
        if success and serve_db_path is not None:
            _copy_database(db_path, serve_db_path, cast("Logger", run_logger))


def _toposort_targets(targets: Iterable[str] | None) -> list[str]:
    desired = list(targets) if targets is not None else ["export_docs"]
    order: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            message = f"Cycle detected in pipeline dependencies at {name}"
            raise RuntimeError(message)
        if name not in PIPELINE_STEPS:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)
        visiting.add(name)
        step = PIPELINE_STEPS[name]
        for dep in step.deps:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for target in desired:
        visit(target)
    return order


def _run_scip_ingest(
    *,
    logger: Logger | LoggerAdapter,
    skip_scip: bool,
    scip_state: dict[str, scip_ingest.ScipIngestResult | None],
    ctx: IngestionContext,
) -> None:
    if skip_scip:
        logger.info("Skipping SCIP ingestion via CODEINTEL_SKIP_SCIP")
        scip_state["result"] = None
        return
    start = _log_task("scip_ingest", logger)
    result = t_scip_ingest.fn(ctx=ctx)
    scip_state["result"] = result
    _log_task_done("scip_ingest", start, logger)
    if result.status != "success":
        logger.info(
            "SCIP ingestion result: %s (%s)",
            result.status,
            result.reason or "no reason",
        )


def _run_symbol_uses(
    *,
    logger: Logger | LoggerAdapter,
    scip_state: dict[str, scip_ingest.ScipIngestResult | None],
    ctx: IngestionContext,
) -> None:
    scip_result = scip_state.get("result")
    if scip_result is not None and scip_result.status != "success":
        logger.info("Skipping symbol_uses: SCIP ingest did not succeed (%s)", scip_result.status)
        return
    scip_json = ctx.build_dir / "scip" / "index.scip.json"
    if not scip_json.is_file():
        logger.info("Skipping symbol_uses: SCIP output unavailable")
        return
    cfg = SymbolUsesConfig.from_paths(
        repo_root=ctx.repo_root,
        scip_json_path=scip_json,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    _run_task("symbol_uses", t_symbol_uses, logger, cfg, ctx.db_path)


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
