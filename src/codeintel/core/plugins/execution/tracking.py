"""Pipeline run tracking helpers for plugin executors.

This module provides shared utilities for recording plugin execution
results to the pipeline run tracking system, extracting common logic
from domain-specific executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from codeintel.storage.tracking import (
    PipelineStepRecord,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from codeintel.core.plugins.types.result import PluginExecutionRecord
    from codeintel.storage.tracking import (
        ModuleKind,
        PipelineRunTracking,
        PipelineStatus,
        StepStatus,
    )


class FatalHandling(Enum):
    """Strategy for handling fatal errors during plugin execution."""

    CONTINUE = "continue"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class TrackingOptions:
    """Options controlling run tracking behavior."""

    fatal_handling: FatalHandling = FatalHandling.CONTINUE


def record_plugin_steps(
    runs: PipelineRunTracking,
    run_id: str,
    module: ModuleKind,
    records: Sequence[PluginExecutionRecord],
    get_stage: Callable[[str], str],
) -> None:
    """Record plugin execution records as pipeline steps.

    Convert plugin execution records to pipeline step records and
    persist them to the run tracking system.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor from gateway.
    run_id
        Run identifier.
    module
        Module name (e.g., "analytics", "graphs", "ingestion").
    records
        Plugin execution records to convert.
    get_stage
        Function to get stage name from plugin name.

    Examples
    --------
    >>> def get_stage(plugin_name: str) -> str:
    ...     return "core" if plugin_name.startswith("core.") else "other"
    >>>
    """
    for rec in records:
        stage = get_stage(rec.plugin_name)

        row_counts: dict[str, int] | None = None
        if rec.result is not None and rec.result.row_counts is not None:
            row_counts = dict(rec.result.row_counts)

        step_status = _map_plugin_status_to_step_status(rec.status)

        extra = _build_step_extra(rec)

        runs.record_step(
            PipelineStepRecord(
                run_id=run_id,
                module=module,
                stage=stage,
                name=rec.plugin_name,
                status=step_status,
                started_at=rec.started_at,
                completed_at=rec.ended_at,
                row_counts=row_counts,
                extra=extra if extra else None,
            ),
        )


def complete_run_from_records(
    runs: PipelineRunTracking,
    run_id: str,
    records: Sequence[PluginExecutionRecord],
    options: TrackingOptions | None = None,
) -> None:
    """Complete a pipeline run based on execution records.

    Analyze execution records to determine overall status and
    complete the pipeline run with appropriate status and error summary.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor from gateway.
    run_id
        Run identifier.
    records
        Plugin execution records.
    options
        Tracking behavior such as fatal error handling.

    Examples
    --------
    >>>
    """
    effective_options = options or TrackingOptions()
    status, error_summary = _compute_run_status(records, effective_options.fatal_handling)

    runs.complete_run(
        run_id,
        status=status,
        error_summary=error_summary,
    )


def _map_plugin_status_to_step_status(plugin_status: str) -> StepStatus:
    """Map plugin execution status to pipeline step status.

    Parameters
    ----------
    plugin_status
        Plugin status string.

    Returns
    -------
    StepStatus
        Corresponding step status.
    """
    if plugin_status == "succeeded":
        return "succeeded"
    if plugin_status == "failed":
        return "failed"
    if plugin_status == "skipped":
        return "skipped"
    return "failed"


def _build_step_extra(rec: PluginExecutionRecord) -> dict[str, object]:
    """Build extra metadata dictionary for a step record.

    Parameters
    ----------
    rec
        Plugin execution record.

    Returns
    -------
    dict[str, object]
        Extra metadata dictionary.
    """
    extra: dict[str, object] = {}
    if rec.error:
        extra["error"] = rec.error
    if rec.partial:
        extra["partial"] = True
    if rec.attempts > 1:
        extra["attempts"] = rec.attempts
    return extra


def _compute_run_status(
    records: Sequence[PluginExecutionRecord],
    fatal_handling: FatalHandling,
) -> tuple[PipelineStatus, str | None]:
    """Compute overall run status from execution records.

    Parameters
    ----------
    records
        Plugin execution records.
    fatal_handling
        Strategy for fatal errors.

    Returns
    -------
    tuple[PipelineStatus, str | None]
        Tuple of (status, error_summary).
    """
    if fatal_handling is FatalHandling.FAIL_FAST:
        failed_plugins = [r.plugin_name for r in records if r.status == "failed"]
        error_summary = f"Fatal error. Failed plugins: {', '.join(failed_plugins)}"
        return "failed", error_summary

    failure_count = sum(1 for r in records if r.status == "failed")
    success_count = sum(1 for r in records if r.status == "succeeded")
    skip_count = sum(1 for r in records if r.status == "skipped")

    if failure_count > 0:
        failed_plugins = [r.plugin_name for r in records if r.status == "failed"]
        error_summary = f"Failed plugins: {', '.join(failed_plugins)}"
        return "failed", error_summary

    if skip_count > 0 and success_count == 0:
        return "partial", None

    return "succeeded", None


__all__ = [
    "FatalHandling",
    "TrackingOptions",
    "complete_run_from_records",
    "record_plugin_steps",
]
