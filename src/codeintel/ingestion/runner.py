"""Centralized ingestion entrypoints sharing a common context."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeintel.config import SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion import change_tracker as change_tracker_module
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.steps import DEFAULT_REGISTRY, IngestStepRegistry
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class IngestionContext:
    """Shared parameters required for all ingestion steps."""

    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    tools: ToolsConfig
    code_profile_cfg: ScanProfile
    config_profile_cfg: ScanProfile
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
        return self.paths.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory resolved for the snapshot."""
        return self.paths.document_output_dir

    @property
    def code_profile(self) -> ScanProfile:
        """Code scanning profile for the run."""
        return self.code_profile_cfg

    @property
    def config_profile(self) -> ScanProfile:
        """Config scanning profile for the run."""
        return self.config_profile_cfg

    @property
    def active_tools(self) -> ToolsConfig:
        """Tools configuration, honoring overrides when provided."""
        return self.tools

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


def _run_ingest_step(
    ctx: IngestionContext,
    name: str,
    *,
    registry: IngestStepRegistry = DEFAULT_REGISTRY,
) -> object | None:
    """
    Run a single ingestion step by name with logging.

    Parameters
    ----------
    ctx
        Shared ingestion context.
    name
        Name of the ingestion step to execute.
    registry
        Registry providing the requested step.

    Returns
    -------
    object | None
        Any value returned by the underlying step.
    """
    start = _log_step_start(name, ctx)
    step = registry.get(name)
    result = step.run(ctx)
    _log_step_done(name, start, ctx)
    return result


def list_ingest_steps(registry: IngestStepRegistry = DEFAULT_REGISTRY) -> list[dict[str, object]]:
    """
    Return machine-readable metadata for all ingestion steps.

    Returns
    -------
    list[dict[str, object]]
        Dictionaries with name, description, produces_tables, and requires.
    """
    return [
        {
            "name": meta.name,
            "description": meta.description,
            "produces_tables": meta.produces_tables,
            "requires": meta.requires,
        }
        for meta in registry.all_metadata()
    ]


def run_ingest_steps(
    ctx: IngestionContext,
    selected_steps: Sequence[str] | None = None,
    *,
    registry: IngestStepRegistry | None = None,
) -> None:
    """
    Run ingestion steps in dependency order.

    Parameters
    ----------
    ctx
        Shared ingestion context.
    selected_steps
        Optional subset of step names to run. If None, all steps are executed
        in the default registry order, respecting declared dependencies.
    registry
        Registry of steps to execute; defaults to the global registry.
    """
    active_registry = registry or DEFAULT_REGISTRY
    if selected_steps is None:
        names = list(active_registry.step_names())
    else:
        expanded = active_registry.expand_with_deps(selected_steps)
        names = active_registry.topological_order(sorted(expanded))
    for name in names:
        _run_ingest_step(ctx, name, registry=active_registry)


def run_repo_scan(ctx: IngestionContext) -> change_tracker_module.ChangeTracker:
    """
    Ingest repository structure and modules using the provided storage gateway.

    Returns
    -------
    change_tracker_module.ChangeTracker
        Tracker populated with module changes.

    Raises
    ------
    RuntimeError
        If the step fails to populate a change tracker.
    """
    _run_ingest_step(ctx, "repo_scan")
    tracker = ctx.change_tracker
    if tracker is None:
        message = "repo_scan step did not populate change_tracker"
        raise RuntimeError(message)
    return tracker


def run_scip_ingest(ctx: IngestionContext) -> ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    This wrapper delegates to the ingestion step registry so that the
    SCIP ingestion logic can live in a pluggable IngestStep implementation.

    Returns
    -------
    ScipIngestResult
        Status and artifact paths for the SCIP run.

    Raises
    ------
    TypeError
        If the registry returns an unexpected result type.
    """
    result = _run_ingest_step(ctx, "scip_ingest")
    if not isinstance(result, ScipIngestResult):
        message = "scip_ingest step returned an unexpected result"
        raise TypeError(message)
    return result


def run_cst_extract(ctx: IngestionContext) -> None:
    """Extract LibCST nodes for the repository using the gateway connection."""
    _run_ingest_step(ctx, "cst_extract")


def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    _run_ingest_step(ctx, "ast_extract")


def run_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines from coverage.json or coverage.py data via the gateway connection."""
    _run_ingest_step(ctx, "coverage_ingest")


def run_tests_ingest(ctx: IngestionContext) -> None:
    """Ingest pytest catalog rows via the gateway connection."""
    _run_ingest_step(ctx, "tests_ingest")


def run_typing_ingest(ctx: IngestionContext) -> None:
    """Collect static typing diagnostics and typedness via the gateway connection."""
    _run_ingest_step(ctx, "typing_ingest")


def run_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract docstrings and persist structured rows via the gateway connection."""
    _run_ingest_step(ctx, "docstrings_ingest")


def run_config_ingest(ctx: IngestionContext) -> None:
    """Flatten configuration files into analytics.config_values via the gateway connection."""
    _run_ingest_step(ctx, "config_ingest")
