"""Centralized ingestion entrypoints sharing a common context."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codeintel.config import ExecutionConfig, ScipIngestStepConfig, SnapshotRef, ToolBinaries
from codeintel.config.models import (
    ConfigIngestConfig,
    CoverageIngestConfig,
    DocstringConfig,
    RepoScanConfig,
    TestsIngestConfig,
    ToolsConfig,
    TypingIngestConfig,
)
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion import change_tracker as change_tracker_module
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
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class IngestionContext:
    """Shared parameters required for all ingestion steps."""

    snapshot: SnapshotRef
    execution: ExecutionConfig
    paths: BuildPaths
    gateway: StorageGateway
    tools: ToolsConfig | None = None
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    change_tracker: change_tracker_module.ChangeTracker | None = None

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot."""
        return self.snapshot.repo_root

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot."""
        return self.snapshot.commit

    @property
    def build_dir(self) -> Path:
        """Build directory derived from execution config."""
        return self.execution.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory resolved for the snapshot."""
        return self.paths.document_output_dir

    @property
    def code_profile(self) -> ScanProfile:
        """Code scanning profile for the run."""
        return self.execution.code_profile

    @property
    def config_profile(self) -> ScanProfile:
        """Config scanning profile for the run."""
        return self.execution.config_profile

    @property
    def active_tools(self) -> ToolsConfig:
        """Tools configuration, honoring overrides when provided."""
        return self.tools or self.execution.tools

    @property
    def db_path(self) -> Path:
        """DuckDB path backing the current gateway."""
        return self.gateway.config.db_path


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


def _require_change_tracker(ctx: IngestionContext) -> change_tracker_module.ChangeTracker:
    """
    Return the change tracker populated during repo_scan.

    Raises
    ------
    RuntimeError
        When the tracker is missing because repo_scan has not been executed.

    Returns
    -------
    change_tracker_module.ChangeTracker
        Cached change tracker for the current ingestion context.
    """
    tracker = ctx.change_tracker
    if tracker is None:
        message = "change_tracker is not set; run repo_scan before incremental ingest"
        raise RuntimeError(message)
    return tracker


def run_repo_scan(ctx: IngestionContext) -> change_tracker_module.ChangeTracker:
    """Ingest repository structure and modules using the provided storage gateway.

    Returns
    -------
    change_tracker_module.ChangeTracker
        Tracker populated with module changes.
    """
    start = _log_step_start("repo_scan", ctx)
    cfg = RepoScanConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tool_runner=ctx.tool_runner,
    )
    tracker = repo_scan.ingest_repo(
        ctx.gateway,
        cfg=cfg,
        code_profile=ctx.code_profile,
    )
    ctx.change_tracker = tracker
    _log_step_done("repo_scan", start, ctx)
    return tracker


def run_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for the SCIP run.
    """
    start = _log_step_start("scip_ingest", ctx)
    binaries = ToolBinaries(
        scip_python_bin=ctx.active_tools.scip_python_bin,
        scip_bin=ctx.active_tools.scip_bin,
        pyright_bin=ctx.active_tools.pyright_bin,
        pyrefly_bin=ctx.active_tools.pyrefly_bin,
        ruff_bin=ctx.active_tools.ruff_bin,
        coverage_bin=ctx.active_tools.coverage_bin,
        pytest_bin=ctx.active_tools.pytest_bin,
        git_bin=ctx.active_tools.git_bin,
        default_timeout_s=ctx.active_tools.default_timeout_s,
    )
    cfg = ScipIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        binaries=binaries,
        scip_runner=ctx.scip_runner,
        artifact_writer=ctx.artifact_writer,
    )
    tracker = ctx.change_tracker
    if tracker is None:
        tracker = _require_change_tracker(ctx)
    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.active_tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.active_tools)
    result = scip_ingest.ingest_scip(
        ctx.gateway,
        cfg=cfg,
        tracker=tracker,
        tool_service=service,
    )
    _log_step_done("scip_ingest", start, ctx)
    return result


def run_cst_extract(ctx: IngestionContext) -> None:
    """Extract LibCST nodes for the repository using the gateway connection."""
    start = _log_step_start("cst_extract", ctx)
    tracker = _require_change_tracker(ctx)
    cst_extract.ingest_cst(
        tracker,
        executor_kind=os.getenv("CODEINTEL_CST_EXECUTOR", "process"),
    )
    _log_step_done("cst_extract", start, ctx)


def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    start = _log_step_start("ast_extract", ctx)
    tracker = _require_change_tracker(ctx)
    py_ast_extract.ingest_python_ast(tracker)
    _log_step_done("ast_extract", start, ctx)


def run_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines from coverage.json or coverage.py data via the gateway connection."""
    start = _log_step_start("coverage_ingest", ctx)
    cfg = CoverageIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        coverage_file=ctx.active_tools.coverage_file,  # type: ignore[arg-type]
        tool_runner=ctx.tool_runner,
    )
    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.active_tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.active_tools)
    coverage_ingest.ingest_coverage_lines(
        gateway=ctx.gateway,
        cfg=cfg,
        tools=ctx.active_tools,
        tool_service=service,
        json_output_path=ctx.paths.coverage_json,
    )
    _log_step_done("coverage_ingest", start, ctx)


def run_tests_ingest(ctx: IngestionContext) -> None:
    """Ingest pytest catalog rows via the gateway connection."""
    start = _log_step_start("tests_ingest", ctx)
    cfg = TestsIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        pytest_report_path=ctx.paths.pytest_report,
    )
    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.active_tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.active_tools)
    tests_ingest.ingest_tests(
        gateway=ctx.gateway,
        cfg=cfg,
        report_path=ctx.paths.pytest_report,
        tools=ctx.active_tools,
        tool_service=service,
    )
    _log_step_done("tests_ingest", start, ctx)


def run_typing_ingest(ctx: IngestionContext) -> None:
    """Collect static typing diagnostics and typedness via the gateway connection."""
    start = _log_step_start("typing_ingest", ctx)
    cfg = TypingIngestConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
        tool_runner=ctx.tool_runner,
    )
    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.active_tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.active_tools)
    typing_ingest.ingest_typing_signals(
        gateway=ctx.gateway,
        cfg=cfg,
        code_profile=ctx.code_profile,
        tools=ctx.active_tools,
        tool_service=service,
    )
    _log_step_done("typing_ingest", start, ctx)


def run_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract docstrings and persist structured rows via the gateway connection."""
    start = _log_step_start("docstrings_ingest", ctx)
    cfg = DocstringConfig(
        repo_root=ctx.repo_root,
        repo=ctx.repo,
        commit=ctx.commit,
    )
    docstrings_ingest.ingest_docstrings(
        ctx.gateway,
        cfg,
        code_profile=ctx.code_profile,
    )
    _log_step_done("docstrings_ingest", start, ctx)


def run_config_ingest(ctx: IngestionContext) -> None:
    """Flatten configuration files into analytics.config_values via the gateway connection."""
    start = _log_step_start("config_ingest", ctx)
    cfg = ConfigIngestConfig(repo_root=ctx.repo_root, repo=ctx.repo, commit=ctx.commit)
    config_ingest.ingest_config_values(ctx.gateway, cfg=cfg, config_profile=ctx.config_profile)
    _log_step_done("config_ingest", start, ctx)
