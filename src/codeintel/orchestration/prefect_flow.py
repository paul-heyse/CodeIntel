"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import logging
import shutil
from datetime import timedelta
from pathlib import Path

import duckdb
from prefect import flow, task
from prefect.tasks import task_input_hash

from codeintel.analytics.ast_metrics import HotspotsConfig, build_hotspots
from codeintel.analytics.coverage_analytics import (
    CoverageAnalyticsConfig,
    compute_coverage_functions,
)
from codeintel.analytics.functions import (
    FunctionAnalyticsConfig,
    compute_function_metrics_and_types,
)
from codeintel.analytics.tests_analytics import (
    TestCoverageConfig,
    compute_test_coverage_edges,
)
from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.graphs.callgraph_builder import CallGraphConfig, build_call_graph
from codeintel.graphs.cfg_builder import CFGBuilderConfig, build_cfg_and_dfg
from codeintel.graphs.goid_builder import GoidBuilderConfig, build_goids
from codeintel.graphs.import_graph import ImportGraphConfig, build_import_graph
from codeintel.graphs.symbol_uses import SymbolUsesConfig, build_symbol_use_edges
from codeintel.ingestion import (
    ast_cst_extract,
    config_ingest,
    coverage_ingest,
    docstrings_ingest,
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

_CACHE_EXPIRATION = timedelta(hours=12)
log = logging.getLogger(__name__)


@task(
    name="repo_scan",
    retries=2,
    retry_delay_seconds=2,
    cache_key_fn=task_input_hash,
    cache_expiration=_CACHE_EXPIRATION,
)
def t_repo_scan(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Ingest repository modules and repo_map rows."""
    con = _connect(db_path)
    repo_scan.ingest_repo(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(
    name="scip_ingest",
    retries=2,
    retry_delay_seconds=5,
    cache_key_fn=task_input_hash,
    cache_expiration=_CACHE_EXPIRATION,
)
def t_scip_ingest(cfg: scip_ingest.ScipIngestConfig, db_path: Path) -> None:
    """Run scip-python ingestion and register SCIP artifacts."""
    con = _connect(db_path)
    scip_ingest.ingest_scip(con=con, cfg=cfg)
    con.close()


@task(
    name="ast_cst",
    retries=2,
    retry_delay_seconds=2,
    cache_key_fn=task_input_hash,
    cache_expiration=_CACHE_EXPIRATION,
)
def t_ast_cst(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Parse AST/CST and persist metrics."""
    con = _connect(db_path)
    ast_cst_extract.ingest_ast_and_cst(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(name="coverage_ingest", retries=1, retry_delay_seconds=2)
def t_coverage_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Load coverage lines into analytics.coverage_lines."""
    con = _connect(db_path)
    coverage_ingest.ingest_coverage_lines(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(name="tests_ingest", retries=1, retry_delay_seconds=2)
def t_tests_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Load pytest test catalog into analytics.test_catalog."""
    con = _connect(db_path)
    tests_ingest.ingest_tests(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(name="typing_ingest", retries=1, retry_delay_seconds=2)
def t_typing_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Collect typedness/static diagnostics."""
    con = _connect(db_path)
    typing_ingest.ingest_typing_signals(con=con, repo_root=repo_root, repo=repo, commit=commit)
    con.close()


@task(
    name="docstrings_ingest",
    retries=1,
    retry_delay_seconds=2,
    cache_key_fn=task_input_hash,
    cache_expiration=_CACHE_EXPIRATION,
)
def t_docstrings_ingest(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Extract structured docstrings into core.docstrings."""
    con = _connect(db_path)
    cfg = docstrings_ingest.DocstringConfig(repo_root=repo_root, repo=repo, commit=commit)
    docstrings_ingest.ingest_docstrings(con, cfg)
    con.close()


@task(name="config_ingest", retries=1, retry_delay_seconds=2)
def t_config_ingest(repo_root: Path, db_path: Path) -> None:
    """Flatten config values into analytics.config_values."""
    con = _connect(db_path)
    config_ingest.ingest_config_values(con=con, repo_root=repo_root)
    con.close()


@task(name="goids", retries=1, retry_delay_seconds=2)
def t_goids(repo: str, commit: str, db_path: Path) -> None:
    """Build GOID registry and crosswalk."""
    con = _connect(db_path)
    cfg = GoidBuilderConfig(repo=repo, commit=commit, language="python")
    build_goids(con, cfg)
    con.close()


@task(name="callgraph", retries=1, retry_delay_seconds=2)
def t_callgraph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Construct call graph nodes and edges."""
    con = _connect(db_path)
    cfg = CallGraphConfig(repo=repo, commit=commit, repo_root=repo_root)
    build_call_graph(con, cfg)
    con.close()


@task(name="cfg_dfg", retries=1, retry_delay_seconds=2)
def t_cfg(repo: str, commit: str, db_path: Path) -> None:
    """Emit minimal CFG/DFG scaffolding."""
    con = _connect(db_path)
    cfg = CFGBuilderConfig(repo=repo, commit=commit)
    build_cfg_and_dfg(con, cfg)
    con.close()


@task(name="import_graph", retries=1, retry_delay_seconds=2)
def t_import_graph(repo: str, commit: str, repo_root: Path, db_path: Path) -> None:
    """Build import graph edges via LibCST analysis."""
    con = _connect(db_path)
    cfg = ImportGraphConfig(repo=repo, commit=commit, repo_root=repo_root)
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
    cfg = HotspotsConfig(repo=repo, commit=commit, repo_root=repo_root)
    build_hotspots(con, cfg)
    con.close()


@task(name="function_metrics", retries=1, retry_delay_seconds=2)
def t_function_metrics(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Compute per-function metrics and types."""
    con = _connect(db_path)
    cfg = FunctionAnalyticsConfig(repo=repo, commit=commit, repo_root=repo_root)
    compute_function_metrics_and_types(con, cfg)
    con.close()


@task(name="coverage_functions", retries=1, retry_delay_seconds=2)
def t_coverage_functions(repo: str, commit: str, db_path: Path) -> None:
    """Aggregate coverage lines to function spans."""
    con = _connect(db_path)
    cfg = CoverageAnalyticsConfig(repo=repo, commit=commit)
    compute_coverage_functions(con, cfg)
    con.close()


@task(name="test_coverage_edges", retries=1, retry_delay_seconds=2)
def t_test_coverage_edges(repo_root: Path, repo: str, commit: str, db_path: Path) -> None:
    """Build test-to-function coverage edges."""
    con = _connect(db_path)
    cfg = TestCoverageConfig(repo=repo, commit=commit, repo_root=repo_root)
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
    document_output: Path,
    scip_cfg: scip_ingest.ScipIngestConfig | None,
    *,
    skip_scip: bool = False,
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
        if scip_cfg is None:
            errors.append("SCIP config missing during preflight")
        if not git_dir.is_dir():
            errors.append("SCIP ingest requires a git repository (.git missing)")
        if scip_cfg is not None:
            if shutil.which(scip_cfg.scip_python_bin) is None:
                errors.append(f"scip-python binary not found: {scip_cfg.scip_python_bin}")
            if shutil.which(scip_cfg.scip_bin) is None:
                errors.append(f"scip binary not found: {scip_cfg.scip_bin}")

    build_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    document_output.mkdir(parents=True, exist_ok=True)

    if errors:
        details = "; ".join(errors)
        message = f"Prefect preflight failed: {details}"
        raise RuntimeError(message)
    log.info("Prefect preflight succeeded for %s@%s", scip_cfg.repo, scip_cfg.commit)


@flow(name="export_docs_flow")
def export_docs_flow(
    repo_root: Path,
    repo: str,
    commit: str,
    db_path: Path,
    build_dir: Path,
    skip_scip: bool = False,
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
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    build_dir = build_dir.resolve()
    document_output = repo_root / "Document Output"

    # Ingestion
    scip_cfg = scip_ingest.ScipIngestConfig(
        repo_root=repo_root,
        repo=repo,
        commit=commit,
        build_dir=build_dir,
        document_output_dir=document_output,
    )
    _preflight_checks(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output=document_output,
        scip_cfg=scip_cfg,
        skip_scip=skip_scip,
    )
    t_repo_scan(repo_root, repo, commit, db_path)
    if not skip_scip:
        t_scip_ingest(cfg=scip_cfg, db_path=db_path)
    t_ast_cst(repo_root, repo, commit, db_path)
    t_coverage_ingest(repo_root, repo, commit, db_path)
    t_tests_ingest(repo_root, repo, commit, db_path)
    t_typing_ingest(repo_root, repo, commit, db_path)
    t_docstrings_ingest(repo_root, repo, commit, db_path)
    t_config_ingest(repo_root, db_path)

    # Graphs
    t_goids(repo, commit, db_path)
    t_callgraph(repo, commit, repo_root, db_path)
    t_cfg(repo, commit, db_path)
    t_import_graph(repo, commit, repo_root, db_path)
    symbol_cfg = SymbolUsesConfig(
        repo_root=repo_root,
        scip_json_path=build_dir / "scip" / "index.scip.json",
        repo=repo,
        commit=commit,
    )
    t_symbol_uses(symbol_cfg, db_path)

    # Analytics
    t_hotspots(repo_root, repo, commit, db_path)
    t_function_metrics(repo_root, repo, commit, db_path)
    t_coverage_functions(repo, commit, db_path)
    t_test_coverage_edges(repo_root, repo, commit, db_path)
    t_risk_factors(repo_root, repo, commit, db_path, build_dir)

    # Export
    t_export_docs(db_path, document_output)
