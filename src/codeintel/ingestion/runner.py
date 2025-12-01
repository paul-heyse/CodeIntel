"""Centralized ingestion entrypoints sharing a common context."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codeintel.config import SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion import change_tracker as change_tracker_module
from codeintel.ingestion.ingest_runs import (
    IngestRun,
    IngestRunMode,
    IngestRunSink,
    IngestRunStatus,
    classify_error,
)
from codeintel.ingestion.scip_ingest import ScipIngestResult
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.steps import DEFAULT_REGISTRY, IngestStepRegistry
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import DuckDBError, StorageGateway

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
    ingest_run_sink: IngestRunSink | None = None
    enable_run_metrics: bool = False
    current_ingest_run: IngestRun | None = None
    step_overrides: Mapping[str, Callable[[IngestionContext], object]] | None = None

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


def _count_rows(gateway: StorageGateway, table_key: str) -> int:
    """
    Return COUNT(*) for a table key, or 0 if the table is missing.

    This is intentionally forgiving: it should never cause a step to fail just because metrics
    are enabled.

    Returns
    -------
    int
        Row count for the requested table when available, otherwise zero.
    """
    try:
        row = gateway.con.table(table_key).aggregate("count(*)").fetchone()
    except DuckDBError:
        return 0
    if row is None:
        return 0
    return int(row[0])


def _guess_run_mode(ctx: IngestionContext, step_name: str) -> IngestRunMode:
    """
    Coarse heuristic for determining run mode.

    - If no change_tracker is present, treat as FULL.
    - For known incremental datasets (those that use run_incremental_ingest), label as
      INCREMENTAL; we do not (yet) distinguish full_rebuild vs true incremental inside the
      harness.

    Returns
    -------
    IngestRunMode
        Resolved mode based on available change-tracker context.
    """
    if ctx.change_tracker is None:
        return IngestRunMode.FULL

    incremental_steps = {
        "ast_extract",
        "cst_extract",
        "scip_ingest",
        "typing_ingest",
        "docstrings_ingest",
    }
    return IngestRunMode.INCREMENTAL if step_name in incremental_steps else IngestRunMode.FULL


def _collect_row_counts(ctx: IngestionContext, datasets: tuple[str, ...]) -> dict[str, int]:
    """
    Collect row counts for the provided dataset tables.

    Returns
    -------
    dict[str, int]
        Mapping of table_key to row count (missing tables return zero).
    """
    if not ctx.enable_run_metrics or not datasets:
        return {}
    counts: dict[str, int] = {}
    for table_key in datasets:
        counts[table_key] = _count_rows(ctx.gateway, table_key)
    return counts


def _compute_row_deltas(
    rows_before: Mapping[str, int],
    rows_after: Mapping[str, int],
) -> tuple[int, int]:
    """
    Compute inserted/deleted deltas from before/after counts.

    Returns
    -------
    tuple[int, int]
        Inserted and deleted row counts in that order.
    """
    inserted = 0
    deleted = 0
    for table_key, after in rows_after.items():
        before = rows_before.get(table_key, 0)
        if after >= before:
            inserted += after - before
        else:
            deleted += before - after
    return inserted, deleted


def _finalize_ingest_run(
    ctx: IngestionContext,
    ingest_run: IngestRun,
    start_ts: float,
    error: BaseException | None,
) -> None:
    """Populate metrics, status, and sinks for a completed ingest run."""
    ingest_run.finished_at = datetime.now(UTC)
    ingest_run.duration_s = time.perf_counter() - start_ts

    rows_after = _collect_row_counts(ctx, ingest_run.datasets)
    ingest_run.rows_after = rows_after

    if rows_after:
        inserted, deleted = _compute_row_deltas(ingest_run.rows_before, rows_after)
        ingest_run.rows_inserted = inserted
        ingest_run.rows_deleted = deleted

    if error is None:
        status = IngestRunStatus.OK
        if (
            ctx.enable_run_metrics
            and ingest_run.mode is IngestRunMode.INCREMENTAL
            and ingest_run.rows_inserted == 0
            and ingest_run.rows_deleted == 0
        ):
            status = IngestRunStatus.SKIPPED
        ingest_run.status = status
        log.info(
            "ingest done: step=%s repo=%s commit=%s run_id=%s status=%s "
            "rows_inserted=%d rows_deleted=%d duration=%.2fs",
            ingest_run.step,
            ingest_run.repo,
            ingest_run.commit,
            ingest_run.run_id,
            ingest_run.status.value,
            ingest_run.rows_inserted,
            ingest_run.rows_deleted,
            ingest_run.duration_s or 0.0,
        )
    else:
        ingest_run.status = IngestRunStatus.ERROR
        ingest_run.error_kind = classify_error(error)
        ingest_run.error_message = str(error)
        log.error(
            "ingest error: step=%s repo=%s commit=%s run_id=%s status=%s error_kind=%s",
            ingest_run.step,
            ingest_run.repo,
            ingest_run.commit,
            ingest_run.run_id,
            ingest_run.status.value,
            ingest_run.error_kind,
        )

    if ctx.ingest_run_sink is not None:
        try:
            ctx.ingest_run_sink.record(ingest_run)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ):  # pragma: no cover - sink errors should not break ingestion
            log.exception(
                "Failed to record ingest run for step=%s run_id=%s",
                ingest_run.step,
                ingest_run.run_id,
            )


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
    step = registry.get(name)
    datasets = tuple(step.produces_tables)
    mode = _guess_run_mode(ctx, name)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    start_ts = time.perf_counter()

    rows_before = _collect_row_counts(ctx, datasets)

    ingest_run = IngestRun(
        run_id=run_id,
        repo=ctx.repo,
        commit=ctx.commit,
        step=name,
        datasets=datasets,
        mode=mode,
        started_at=started_at,
        rows_before=rows_before,
    )
    ctx.current_ingest_run = ingest_run

    log.info(
        "ingest start: step=%s repo=%s commit=%s run_id=%s",
        name,
        ctx.repo,
        ctx.commit,
        run_id,
    )

    error: BaseException | None = None
    result: object | None = None

    runner = None
    if ctx.step_overrides is not None:
        runner = ctx.step_overrides.get(name)

    try:
        result = runner(ctx) if runner is not None else step.run(ctx)
    except BaseException as exc:
        error = exc
        raise
    finally:
        _finalize_ingest_run(ctx, ingest_run, start_ts, error)
        ctx.current_ingest_run = None

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
