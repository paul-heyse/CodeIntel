"""Centralized ingestion entrypoints sharing a common context."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from codeintel.config.models import (
    ConfigIngestConfig,
    CoverageIngestConfig,
    DocstringConfig,
    PathsConfig,
    PyAstIngestConfig,
    RepoScanConfig,
    ScipIngestConfig,
    TestsIngestConfig,
    ToolsConfig,
    TypingIngestConfig,
)
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
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.ingestion.tool_runner import ToolRunner

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildPaths:
    """Standardized build artifacts used across ingestion steps."""

    build_dir: Path

    @property
    def tool_cache(self) -> Path:
        """Directory for tool cache outputs."""
        return (self.build_dir / ".tool_cache").resolve()

    @property
    def coverage_json(self) -> Path:
        """Default path for coverage CLI JSON output."""
        return (self.build_dir / "coverage.json").resolve()

    @property
    def pytest_report(self) -> Path:
        """Default path for pytest JSON report."""
        return (self.build_dir / "pytest-report.json").resolve()


@dataclass(frozen=True)
class IngestionContext:
    """Shared parameters required for all ingestion steps."""

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    document_output_dir: Path
    tools: ToolsConfig | None = None
    scan_config: ScanConfig | None = None
    tool_runner: ToolRunner | None = None

    @property
    def paths(self) -> PathsConfig:
        """PathsConfig derived from the current context."""
        return PathsConfig(
            repo_root=self.repo_root,
            build_dir=self.build_dir,
            db_path=self.db_path,
            document_output_dir=self.document_output_dir,
        )

    @property
    def build_paths(self) -> BuildPaths:
        """Helper for common build/output locations."""
        return BuildPaths(build_dir=self.build_dir)


def _log_step_start(step: str, ctx: IngestionContext) -> float:
    """
    Emit a start log for an ingestion step.

    Returns
    -------
    float
        Start timestamp for duration tracking.
    """
    log.info("ingest start: %s repo=%s commit=%s", step, ctx.repo, ctx.commit)
    return time.perf_counter()


def _log_step_done(step: str, start_ts: float, ctx: IngestionContext) -> None:
    """Emit completion log with duration for an ingestion step."""
    duration = time.perf_counter() - start_ts
    log.info("ingest done: %s repo=%s commit=%s (%.2fs)", step, ctx.repo, ctx.commit, duration)


def run_repo_scan(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Ingest repository structure and modules."""
    start = _log_step_start("repo_scan", ctx)
    cfg = RepoScanConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    repo_scan.ingest_repo(con=con, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("repo_scan", start, ctx)


def run_scip_ingest(
    con: duckdb.DuckDBPyConnection, ctx: IngestionContext
) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for the SCIP run.
    """
    start = _log_step_start("scip_ingest", ctx)
    cfg = ScipIngestConfig.from_paths(
        repo=ctx.repo,
        commit=ctx.commit,
        paths=ctx.paths,
        tools=ctx.tools,
    )
    result = scip_ingest.ingest_scip(con=con, cfg=cfg)
    _log_step_done("scip_ingest", start, ctx)
    return result


def run_cst_extract(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Extract LibCST nodes for the repository."""
    start = _log_step_start("cst_extract", ctx)
    cst_extract.ingest_cst(
        con=con,
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        scan_config=ctx.scan_config,
    )
    _log_step_done("cst_extract", start, ctx)


def run_ast_extract(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics."""
    start = _log_step_start("ast_extract", ctx)
    cfg = PyAstIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    py_ast_extract.ingest_python_ast(con=con, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("ast_extract", start, ctx)


def run_coverage_ingest(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Load coverage lines from coverage.json or coverage.py data."""
    start = _log_step_start("coverage_ingest", ctx)
    cfg = CoverageIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tools=ctx.tools,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    coverage_ingest.ingest_coverage_lines(
        con=con,
        cfg=cfg,
        runner=runner,
        json_output_path=ctx.build_paths.coverage_json,
    )
    _log_step_done("coverage_ingest", start, ctx)


def run_tests_ingest(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Ingest pytest catalog rows."""
    start = _log_step_start("tests_ingest", ctx)
    cfg = TestsIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tools=ctx.tools,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    tests_ingest.ingest_tests(
        con=con,
        cfg=cfg,
        runner=runner,
        report_path=ctx.build_paths.pytest_report,
    )
    _log_step_done("tests_ingest", start, ctx)


def run_typing_ingest(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Collect static typing diagnostics and typedness."""
    start = _log_step_start("typing_ingest", ctx)
    cfg = TypingIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    typing_ingest.ingest_typing_signals(
        con=con,
        cfg=cfg,
        scan_config=ctx.scan_config,
        runner=runner,
        tools=ctx.tools,
    )
    _log_step_done("typing_ingest", start, ctx)


def run_docstrings_ingest(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Extract docstrings and persist structured rows."""
    start = _log_step_start("docstrings_ingest", ctx)
    cfg = DocstringConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    docstrings_ingest.ingest_docstrings(con, cfg, scan_config=ctx.scan_config)
    _log_step_done("docstrings_ingest", start, ctx)


def run_config_ingest(con: duckdb.DuckDBPyConnection, ctx: IngestionContext) -> None:
    """Flatten configuration files into analytics.config_values."""
    start = _log_step_start("config_ingest", ctx)
    cfg = ConfigIngestConfig.from_paths(repo_root=ctx.repo_root)
    config_ingest.ingest_config_values(con=con, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("config_ingest", start, ctx)
