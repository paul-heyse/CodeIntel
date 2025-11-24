"""Centralized ingestion entrypoints sharing a common context."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.storage.gateway import StorageGateway

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
        return (self.build_dir / "test-results" / "pytest-report.json").resolve()


@dataclass(frozen=True)
class IngestionContext:
    """Shared parameters required for all ingestion steps."""

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    document_output_dir: Path
    gateway: StorageGateway
    tools: ToolsConfig | None = None
    scan_config: ScanConfig | None = None
    tool_runner: ToolRunner | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None

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


def run_repo_scan(ctx: IngestionContext) -> None:
    """Ingest repository structure and modules using the provided storage gateway."""
    start = _log_step_start("repo_scan", ctx)
    cfg = RepoScanConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tool_runner=ctx.tool_runner,
    )
    repo_scan.ingest_repo(ctx.gateway, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("repo_scan", start, ctx)


def run_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for the SCIP run.
    """
    start = _log_step_start("scip_ingest", ctx)
    doc_dir = ctx.paths.document_output_dir or (ctx.repo_root / "Document Output")
    cfg = ScipIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        build_dir=ctx.build_paths.build_dir,
        document_output_dir=doc_dir,
        scip_python_bin=ctx.tools.scip_python_bin if ctx.tools else "scip-python",
        scip_bin=ctx.tools.scip_bin if ctx.tools else "scip",
        scip_runner=ctx.scip_runner,
        artifact_writer=ctx.artifact_writer,
    )
    result = scip_ingest.ingest_scip(ctx.gateway, cfg=cfg)
    _log_step_done("scip_ingest", start, ctx)
    return result


def run_cst_extract(ctx: IngestionContext) -> None:
    """Extract LibCST nodes for the repository using the gateway connection."""
    start = _log_step_start("cst_extract", ctx)
    cst_extract.ingest_cst(
        ctx.gateway,
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        scan_config=ctx.scan_config,
    )
    _log_step_done("cst_extract", start, ctx)


def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    start = _log_step_start("ast_extract", ctx)
    cfg = PyAstIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    py_ast_extract.ingest_python_ast(ctx.gateway, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("ast_extract", start, ctx)


def run_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines from coverage.json or coverage.py data via the gateway connection."""
    start = _log_step_start("coverage_ingest", ctx)
    cfg = CoverageIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        coverage_file=(ctx.tools.coverage_file if ctx.tools else None),  # type: ignore[arg-type]
        tool_runner=ctx.tool_runner,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    coverage_ingest.ingest_coverage_lines(
        gateway=ctx.gateway,
        cfg=cfg,
        runner=runner,
        json_output_path=ctx.build_paths.coverage_json,
    )
    _log_step_done("coverage_ingest", start, ctx)


def run_tests_ingest(ctx: IngestionContext) -> None:
    """Ingest pytest catalog rows via the gateway connection."""
    start = _log_step_start("tests_ingest", ctx)
    cfg = TestsIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tools=ctx.tools,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    tests_ingest.ingest_tests(
        gateway=ctx.gateway,
        cfg=cfg,
        runner=runner,
        report_path=ctx.build_paths.pytest_report,
    )
    _log_step_done("tests_ingest", start, ctx)


def run_typing_ingest(ctx: IngestionContext) -> None:
    """Collect static typing diagnostics and typedness via the gateway connection."""
    start = _log_step_start("typing_ingest", ctx)
    cfg = TypingIngestConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tool_runner=ctx.tool_runner,
    )
    runner = ctx.tool_runner or ToolRunner(cache_dir=ctx.build_paths.tool_cache)
    typing_ingest.ingest_typing_signals(
        gateway=ctx.gateway,
        cfg=cfg,
        scan_config=ctx.scan_config,
        runner=runner,
        tools=ctx.tools,
    )
    _log_step_done("typing_ingest", start, ctx)


def run_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract docstrings and persist structured rows via the gateway connection."""
    start = _log_step_start("docstrings_ingest", ctx)
    cfg = DocstringConfig.from_paths(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    docstrings_ingest.ingest_docstrings(ctx.gateway, cfg, scan_config=ctx.scan_config)
    _log_step_done("docstrings_ingest", start, ctx)


def run_config_ingest(ctx: IngestionContext) -> None:
    """Flatten configuration files into analytics.config_values via the gateway connection."""
    start = _log_step_start("config_ingest", ctx)
    cfg = ConfigIngestConfig.from_paths(repo_root=ctx.repo_root)
    config_ingest.ingest_config_values(ctx.gateway, cfg=cfg, scan_config=ctx.scan_config)
    _log_step_done("config_ingest", start, ctx)
