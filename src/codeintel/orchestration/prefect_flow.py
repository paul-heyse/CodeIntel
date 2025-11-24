"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from pathlib import Path

import duckdb
from prefect import flow, get_run_logger, task

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.functions import compute_function_metrics_and_types
from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.tests_analytics import compute_test_coverage_edges
from codeintel.config.models import (
    CallGraphConfig,
    CFGBuilderConfig,
    CoverageAnalyticsConfig,
    FunctionAnalyticsConfig,
    FunctionAnalyticsOverrides,
    GoidBuilderConfig,
    GraphMetricsConfig,
    HotspotsConfig,
    ImportGraphConfig,
    SubsystemsConfig,
    SymbolUsesConfig,
    TestCoverageConfig,
    ToolsConfig,
)
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
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
from codeintel.ingestion.source_scanner import IGNORES, ScanConfig
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.orchestration.steps import (
    PIPELINE_STEPS,
    PipelineContext,
    ProfilesStep,
    RiskFactorsStep,
)
from codeintel.storage.views import create_all_views


@dataclass(frozen=True)
class ExportArgs:
    """Inputs for the export_docs Prefect flow."""

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    tools: ToolsConfig | None = None
    scan_config: ScanConfig | None = None
    function_overrides: FunctionAnalyticsOverrides | None = None


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
        "CODEINTEL_COVERAGE_FILE": "coverage_file",
        "CODEINTEL_PYTEST_REPORT": "pytest_report_path",
    }
    for env_var, field in env_map.items():
        value = os.getenv(env_var)
        if value:
            data[field] = value
    return ToolsConfig.model_validate(data)


def _scan_from_env(repo_root: Path, base: ScanConfig | None = None) -> ScanConfig:
    """
    Build a ScanConfig with optional environment overrides for ignore/include patterns.

    Returns
    -------
    ScanConfig
        Normalized scan configuration for source discovery.
    """
    include_patterns = base.include_patterns if base is not None else ("*.py",)
    ignore_dirs = tuple(base.ignore_dirs) if base is not None else tuple(sorted(IGNORES))
    log_every = base.log_every if base is not None else 50
    log_interval = base.log_interval if base is not None else 5.0

    env_includes = os.getenv("CODEINTEL_INCLUDE_PATTERNS")
    if env_includes:
        include_patterns = tuple(
            pattern.strip() for pattern in env_includes.split(",") if pattern.strip()
        )
    env_ignores = os.getenv("CODEINTEL_IGNORE_DIRS")
    if env_ignores:
        merged = set(ignore_dirs) | {
            part.strip() for part in env_ignores.split(",") if part.strip()
        }
        ignore_dirs = tuple(sorted(merged))

    return ScanConfig(
        repo_root=repo_root,
        include_patterns=include_patterns,
        ignore_dirs=ignore_dirs,
        log_every=log_every,
        log_interval=log_interval,
    )


@task(
    name="repo_scan",
    retries=2,
    retry_delay_seconds=2,
)
def t_repo_scan(ctx: IngestionContext) -> None:
    """Ingest repository modules and repo_map rows."""
    con = _connect(ctx.db_path)
    run_repo_scan(con, ctx)
    con.close()


@task(
    name="scip_ingest",
    retries=2,
    retry_delay_seconds=5,
)
def t_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Run scip-python ingestion and register SCIP artifacts.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for SCIP ingestion.
    """
    con = _connect(ctx.db_path)
    result = run_scip_ingest(con, ctx)
    if result.status != "success":
        log.info("SCIP ingestion skipped or failed: %s", result.reason or result.status)
    con.close()
    return result


@task(
    name="cst_extract",
    retries=2,
    retry_delay_seconds=2,
)
def t_cst_extract(ctx: IngestionContext) -> None:
    """Parse CST and persist into cst_nodes."""
    con = _connect(ctx.db_path)
    run_cst_extract(con, ctx)
    con.close()


@task(
    name="ast_extract",
    retries=2,
    retry_delay_seconds=2,
)
def t_ast_extract(ctx: IngestionContext) -> None:
    """Parse Python stdlib AST and persist into ast tables."""
    con = _connect(ctx.db_path)
    run_ast_extract(con, ctx)
    con.close()


@task(name="coverage_ingest", retries=1, retry_delay_seconds=2)
def t_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines into analytics.coverage_lines."""
    con = _connect(ctx.db_path)
    run_coverage_ingest(con, ctx)
    con.close()


@task(name="tests_ingest", retries=1, retry_delay_seconds=2)
def t_tests_ingest(ctx: IngestionContext) -> None:
    """Load pytest test catalog into analytics.test_catalog."""
    con = _connect(ctx.db_path)
    run_tests_ingest(con, ctx)
    con.close()


@task(name="typing_ingest", retries=1, retry_delay_seconds=2)
def t_typing_ingest(ctx: IngestionContext) -> None:
    """Collect typedness/static diagnostics."""
    con = _connect(ctx.db_path)
    run_typing_ingest(con, ctx)
    con.close()


@task(
    name="docstrings_ingest",
    retries=1,
    retry_delay_seconds=2,
)
def t_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract structured docstrings into core.docstrings."""
    con = _connect(ctx.db_path)
    run_docstrings_ingest(con, ctx)
    con.close()


@task(name="config_ingest", retries=1, retry_delay_seconds=2)
def t_config_ingest(ctx: IngestionContext) -> None:
    """Flatten config values into analytics.config_values."""
    con = _connect(ctx.db_path)
    run_config_ingest(con, ctx)
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
def t_hotspots(
    repo_root: Path, repo: str, commit: str, db_path: Path, runner: ToolRunner | None = None
) -> None:
    """Compute file-level hotspot scores."""
    con = _connect(db_path)
    cfg = HotspotsConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit)
    build_hotspots(con, cfg, runner=runner)
    con.close()


@task(name="function_metrics", retries=1, retry_delay_seconds=2)
def t_function_metrics(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    overrides: FunctionAnalyticsOverrides | None = None,
) -> None:
    """Compute per-function metrics and types."""
    run_logger = get_run_logger()
    con = _connect(db_path)
    cfg = FunctionAnalyticsConfig.from_paths(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        overrides=overrides,
    )
    summary = compute_function_metrics_and_types(con, cfg)
    run_logger.info(
        "function_metrics summary rows=%d types=%d validation=%d parse_failed=%d span_not_found=%d",
        summary["metrics_rows"],
        summary["types_rows"],
        summary["validation_total"],
        summary["validation_parse_failed"],
        summary["validation_span_not_found"],
    )
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


@task(name="graph_metrics", retries=1, retry_delay_seconds=2)
def t_graph_metrics(repo: str, commit: str, db_path: Path) -> None:
    """Compute graph metrics for functions and modules."""
    con = _connect(db_path)
    cfg = GraphMetricsConfig.from_paths(repo=repo, commit=commit)
    compute_graph_metrics(con, cfg)
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


@task(name="subsystems", retries=1, retry_delay_seconds=2)
def t_subsystems(repo: str, commit: str, db_path: Path) -> None:
    """Infer subsystem clusters and membership."""
    con = _connect(db_path)
    cfg = SubsystemsConfig.from_paths(repo=repo, commit=commit)
    build_subsystems(con, cfg)
    con.close()


@task(name="profiles", retries=1, retry_delay_seconds=2)
def t_profiles(repo_root: Path, repo: str, commit: str, db_path: Path, build_dir: Path) -> None:
    """Build function, file, and module profiles."""
    con = _connect(db_path)
    step = ProfilesStep()
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


@task(name="graph_validation", retries=1, retry_delay_seconds=2)
def t_graph_validation(repo: str, commit: str, db_path: Path) -> None:
    """Run graph validation warnings."""
    con = _connect(db_path, read_only=False)
    run_graph_validations(con, repo=repo, commit=commit, logger=log)
    con.close()


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


@flow(name="export_docs_flow")
def export_docs_flow(
    args: ExportArgs,
    targets: Iterable[str] | None = None,
) -> None:
    """
    Run the CodeIntel pipeline using Prefect orchestration.

    Parameters
    ----------
    args:
        Repository identity and path configuration.
    targets:
        Optional subset of steps to run (in declared order).
    """
    run_logger = get_run_logger()
    repo_root = args.repo_root.resolve()
    db_path = args.db_path.resolve()
    build_dir = args.build_dir.resolve()
    tools_cfg = _tools_from_env(args.tools)
    scan_cfg = _scan_from_env(repo_root, args.scan_config)
    tool_runner = ToolRunner(cache_dir=build_dir / ".tool_cache")
    skip_scip = os.getenv("CODEINTEL_SKIP_SCIP", "false").lower() == "true"

    ctx = IngestionContext(
        repo_root=repo_root,
        repo=args.repo,
        commit=args.commit,
        db_path=db_path,
        build_dir=build_dir,
        document_output_dir=_resolve_output_dir(repo_root),
        tools=tools_cfg,
        scan_config=scan_cfg,
        tool_runner=tool_runner,
    )
    _preflight_checks(ctx, skip_scip=skip_scip, logger=run_logger)
    scip_state: dict[str, scip_ingest.ScipIngestResult | None] = {"result": None}

    step_handlers = [
        ("repo_scan", lambda: _run_task("repo_scan", t_repo_scan, run_logger, ctx)),
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
        ("ast_extract", lambda: _run_task("ast_extract", t_ast_extract, run_logger, ctx)),
        (
            "coverage_ingest",
            lambda: _run_task("coverage_ingest", t_coverage_ingest, run_logger, ctx),
        ),
        ("tests_ingest", lambda: _run_task("tests_ingest", t_tests_ingest, run_logger, ctx)),
        ("typing_ingest", lambda: _run_task("typing_ingest", t_typing_ingest, run_logger, ctx)),
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
            ),
        ),
        (
            "export_docs",
            lambda: _run_task(
                "export_docs", t_export_docs, run_logger, ctx.db_path, ctx.document_output_dir
            ),
        ),
    ]

    handlers = dict(step_handlers)
    for name in _toposort_targets(targets):
        handler = handlers.get(name)
        if handler is None:
            run_logger.warning("No handler registered for step %s; skipping.", name)
            continue
        handler()


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
