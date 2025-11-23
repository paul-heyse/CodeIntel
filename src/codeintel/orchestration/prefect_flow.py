"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from logging import Logger, LoggerAdapter
from pathlib import Path

import duckdb
from prefect import flow, get_run_logger, task

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.functions import compute_function_metrics_and_types
from codeintel.analytics.tests_analytics import compute_test_coverage_edges
from codeintel.config.models import (
    CallGraphConfig,
    CFGBuilderConfig,
    ConfigIngestConfig,
    CoverageAnalyticsConfig,
    CoverageIngestConfig,
    DocstringConfig,
    FunctionAnalyticsConfig,
    GoidBuilderConfig,
    HotspotsConfig,
    ImportGraphConfig,
    PathsConfig,
    PyAstIngestConfig,
    RepoScanConfig,
    ScipIngestConfig,
    SymbolUsesConfig,
    TestCoverageConfig,
    TestsIngestConfig,
    TypingIngestConfig,
)
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.goid_builder import build_goids
from codeintel.graphs.import_graph import build_import_graph
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.ingestion import (
    config_ingest,
    coverage_ingest,
    cst_extract,
    docstrings_ingest,
    py_ast_extract,
    repo_scan,
    scip_ingest,
    tests_ingest,
    typing_ingest,
)
from codeintel.orchestration.steps import PipelineContext, RiskFactorsStep
from codeintel.storage.views import create_all_views


def _connect(db_path: Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection with an explicit read-only toggle.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Live connection for the requested db_path.
    """
    return duckdb.connect(str(db_path), read_only=read_only)


# ---------------------------------------------------------------------------
# Tasks with light caching/retries for deterministic steps
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


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


def _run_task(
    name: str,
    fn: Callable[..., object],
    logger: Logger | LoggerAdapter,
    *args: object,
    **kwargs: object,
) -> None:
    """Run a Prefect task with start/finish logging."""
    start = _log_task(name, logger)
    fn(*args, **kwargs)
    _log_task_done(name, start, logger)


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


@task(
    name="repo_scan",
    retries=2,
    retry_delay_seconds=2,
)
def t_repo_scan(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Ingest repository modules and repo_map rows."""
    con = _connect(db_path)
    cfg = RepoScanConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    repo_scan.ingest_repo(con=con, cfg=cfg)
    con.close()


@task(
    name="scip_ingest",
    retries=2,
    retry_delay_seconds=5,
)
def t_scip_ingest(cfg: ScipIngestConfig, db_path: Path) -> scip_ingest.ScipIngestResult:
    """
    Run scip-python ingestion and register SCIP artifacts.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for SCIP ingestion.
    """
    con = _connect(db_path)
    result = scip_ingest.ingest_scip(con=con, cfg=cfg)
    if result.status != "success":
        log.info("SCIP ingestion skipped or failed: %s", result.reason or result.status)
    con.close()
    return result


@task(
    name="cst_extract",
    retries=2,
    retry_delay_seconds=2,
)
def t_cst_extract(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Parse CST and persist into cst_nodes."""
    con = _connect(db_path)
    cst_extract.ingest_cst(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(
    name="ast_extract",
    retries=2,
    retry_delay_seconds=2,
)
def t_ast_extract(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Parse Python stdlib AST and persist into ast tables."""
    con = _connect(db_path)
    cfg = PyAstIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    py_ast_extract.ingest_python_ast(con=con, cfg=cfg)
    con.close()


@task(name="coverage_ingest", retries=1, retry_delay_seconds=2)
def t_coverage_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Load coverage lines into analytics.coverage_lines."""
    con = _connect(db_path)
    cfg = CoverageIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    coverage_ingest.ingest_coverage_lines(con=con, cfg=cfg)
    con.close()


@task(name="tests_ingest", retries=1, retry_delay_seconds=2)
def t_tests_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Load pytest test catalog into analytics.test_catalog."""
    con = _connect(db_path)
    cfg = TestsIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    tests_ingest.ingest_tests(con=con, cfg=cfg)
    con.close()


@task(name="typing_ingest", retries=1, retry_delay_seconds=2)
def t_typing_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Collect typedness/static diagnostics."""
    con = _connect(db_path)
    cfg = TypingIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    typing_ingest.ingest_typing_signals(con=con, cfg=cfg)
    con.close()


@task(
    name="docstrings_ingest",
    retries=1,
    retry_delay_seconds=2,
)
def t_docstrings_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Extract structured docstrings into core.docstrings."""
    con = _connect(db_path)
    cfg = DocstringConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    docstrings_ingest.ingest_docstrings(con, cfg)
    con.close()


@task(name="config_ingest", retries=1, retry_delay_seconds=2)
def t_config_ingest(repo_root: Path, db_path: Path) -> None:
    """Flatten config values into analytics.config_values."""
    con = _connect(db_path)
    cfg = ConfigIngestConfig.from_paths(repo_root=repo_root)
    config_ingest.ingest_config_values(con=con, cfg=cfg)
    con.close()


@task(name="goids", retries=1, retry_delay_seconds=2)
def t_goids(repo: str, commit: str, db_path: Path) -> None:
    """Build GOID registry and crosswalk."""
    con = _connect(db_path)
    cfg = GoidBuilderConfig.from_paths(repo=repo, commit=commit, language="python")
    build_goids(con, cfg)
    con.close()


@task(name="callgraph", retries=1, retry_delay_seconds=2)
def t_callgraph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Construct call graph nodes and edges."""
    con = _connect(db_path)
    cfg = CallGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_call_graph(con, cfg)
    con.close()


@task(name="cfg_dfg", retries=1, retry_delay_seconds=2)
def t_cfg(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Emit minimal CFG/DFG scaffolding."""
    con = _connect(db_path)
    cfg = CFGBuilderConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_cfg_and_dfg(con, cfg)
    con.close()


@task(name="import_graph", retries=1, retry_delay_seconds=2)
def t_import_graph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Build import graph edges via LibCST analysis."""
    con = _connect(db_path)
    cfg = ImportGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_import_graph(con, cfg)
    con.close()


@task(name="symbol_uses", retries=1, retry_delay_seconds=2)
def t_symbol_uses(
    cfg: SymbolUsesConfig,
    db_path: Path,
) -> None:
    """Derive symbol use edges from SCIP JSON."""
    con = _connect(db_path)
    build_symbol_use_edges(con, cfg)
    con.close()


@task(name="hotspots", retries=1, retry_delay_seconds=2)
def t_hotspots(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Compute file-level hotspot scores."""
    con = _connect(db_path)
    cfg = HotspotsConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    build_hotspots(con, cfg)
    con.close()


@task(name="function_metrics", retries=1, retry_delay_seconds=2)
def t_function_metrics(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Compute per-function metrics and types."""
    con = _connect(db_path)
    cfg = FunctionAnalyticsConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_metrics_and_types(con, cfg)
    con.close()


@task(name="coverage_functions", retries=1, retry_delay_seconds=2)
def t_coverage_functions(repo: str, commit: str, db_path: Path) -> None:
    """Aggregate coverage lines to function spans."""
    con = _connect(db_path)
    cfg = CoverageAnalyticsConfig.from_paths(repo=repo, commit=commit)
    compute_coverage_functions(con, cfg)
    con.close()


@task(name="test_coverage_edges", retries=1, retry_delay_seconds=2)
def t_test_coverage_edges(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Build test-to-function coverage edges."""
    con = _connect(db_path)
    cfg = TestCoverageConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    compute_test_coverage_edges(con, cfg)
    con.close()


@task(name="risk_factors", retries=1, retry_delay_seconds=2)
def t_risk_factors(repo_root: Path, repo: str, commit: str, db_path: Path, build_dir: Path) -> None:
    """Populate analytics.goid_risk_factors from analytics tables."""
    con = _connect(db_path)
    step = RiskFactorsStep()
    ctx = PipelineContext(
        repo_root=repo_root,
        db_path=db_path,
        build_dir=build_dir,
        repo=repo,
        commit=commit,
    )
    step.run(ctx, con)
    con.close()


@task(name="export_docs", retries=1, retry_delay_seconds=2)
def t_export_docs(db_path: Path, document_output_dir: Path) -> None:
    """Create views and export Parquet/JSONL artifacts."""
    con = _connect(db_path)
    create_all_views(con)
    export_all_parquet(con, document_output_dir)
    export_all_jsonl(con, document_output_dir)
    con.close()


def _preflight_checks(
    repo_root: Path,
    build_dir: Path,
    db_path: Path,
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

    if not repo_root.is_dir():
        errors.append(f"repo_root not found: {repo_root}")
    git_dir = repo_root / ".git"
    if not skip_scip:
        scip_python_bin = "scip-python"
        scip_bin = "scip"
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


@flow(name="export_docs_flow")
def export_docs_flow(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    build_dir: Path,
) -> None:
    """
    Run the CodeIntel pipeline using Prefect orchestration.

    Parameters
    ----------
    repo_root:
        Path to the repository root.
    repo:
        Repository slug (org/repo).
    commit:
        Commit SHA.
    db_path:
        Path to DuckDB database file.
    build_dir:
        Build directory containing intermediates (scip/, etc.).
    """
    run_logger = get_run_logger()
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    build_dir = build_dir.resolve()
    document_output = _resolve_output_dir(repo_root)
    skip_scip = os.getenv("CODEINTEL_SKIP_SCIP", "false").lower() == "true"

    # Ingestion
    paths_cfg = PathsConfig(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=document_output,
    )
    scip_cfg = ScipIngestConfig.from_paths(
        repo=repo,
        commit=commit,
        paths=paths_cfg,
    )
    _preflight_checks(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        skip_scip=skip_scip,
        logger=run_logger,
    )
    _run_task("repo_scan", t_repo_scan, run_logger, repo_root, repo, commit, db_path)
    scip_result: scip_ingest.ScipIngestResult | None = None
    if not skip_scip:
        start = _log_task("scip_ingest", run_logger)
        scip_result = t_scip_ingest.fn(cfg=scip_cfg, db_path=db_path)
        _log_task_done("scip_ingest", start, run_logger)
        if scip_result.status != "success":
            run_logger.info(
                "SCIP ingestion result: %s (%s)",
                scip_result.status,
                scip_result.reason or "no reason",
            )
    else:
        run_logger.info("Skipping SCIP ingestion via CODEINTEL_SKIP_SCIP")
    _run_task("cst_extract", t_cst_extract, run_logger, repo_root, repo, commit, db_path)
    _run_task("ast_extract", t_ast_extract, run_logger, repo_root, repo, commit, db_path)
    _run_task("coverage_ingest", t_coverage_ingest, run_logger, repo_root, repo, commit, db_path)
    _run_task("tests_ingest", t_tests_ingest, run_logger, repo_root, repo, commit, db_path)
    _run_task("typing_ingest", t_typing_ingest, run_logger, repo_root, repo, commit, db_path)
    _run_task(
        "docstrings_ingest", t_docstrings_ingest, run_logger, repo_root, repo, commit, db_path
    )
    _run_task("config_ingest", t_config_ingest, run_logger, repo_root, db_path)

    # Graphs
    _run_task("goids", t_goids, run_logger, repo, commit, db_path)
    _run_task("callgraph", t_callgraph, run_logger, repo, commit, repo_root, db_path)
    _run_task("cfg_dfg", t_cfg, run_logger, repo, commit, repo_root, db_path)
    _run_task("import_graph", t_import_graph, run_logger, repo, commit, repo_root, db_path)
    symbol_cfg = SymbolUsesConfig.from_paths(
        repo_root=repo_root,
        scip_json_path=build_dir / "scip" / "index.scip.json",
        repo=repo,
        commit=commit,
    )
    if (
        scip_result is None or scip_result.status == "success"
    ) and symbol_cfg.scip_json_path.is_file():
        _run_task("symbol_uses", t_symbol_uses, run_logger, symbol_cfg, db_path)
    else:
        run_logger.info("Skipping symbol_uses: SCIP output unavailable")

    # Analytics
    _run_task("hotspots", t_hotspots, run_logger, repo_root, repo, commit, db_path)
    _run_task("function_metrics", t_function_metrics, run_logger, repo_root, repo, commit, db_path)
    _run_task("coverage_functions", t_coverage_functions, run_logger, repo, commit, db_path)
    _run_task(
        "test_coverage_edges", t_test_coverage_edges, run_logger, repo_root, repo, commit, db_path
    )
    _run_task(
        "risk_factors", t_risk_factors, run_logger, repo_root, repo, commit, db_path, build_dir
    )

    # Export
    _run_task("export_docs", t_export_docs, run_logger, db_path, document_output)
