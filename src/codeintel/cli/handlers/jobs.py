"""Background job management handlers.

Handlers for job listing, status, and management operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import ActionResult, JobInfo, JobOutputResult, ListResult
from codeintel.cli.errors.results import (
    fail_job_cancel_failed,
    fail_job_not_completed,
    fail_job_not_found,
)
from codeintel.cli.jobs import JobStatus

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


def jobs_list_handler(ctx: CommandContext) -> CliResult[ListResult[JobInfo]]:
    """List background jobs.

    Parameters
    ----------
    ctx
        Command context with params:
        - status: Optional status filter
        - limit: Maximum jobs to return (default 20)

    Returns
    -------
    CliResult[ListResult[JobInfo]]
        List of jobs.
    """
    status_str = ctx.params.get_str("status")
    limit = ctx.params.get_int("limit", 20)

    LOG.info("Listing jobs with status=%s, limit=%d", status_str, limit)

    status_filter = JobStatus(status_str) if status_str else None
    jobs = ctx.jobs.list_jobs(status=status_filter, limit=limit)

    items = [
        JobInfo(
            job_id=job.job_id,
            operation_id=job.operation_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )
        for job in jobs
    ]

    return CliResult.ok(ListResult.from_items(items))


def jobs_status_handler(ctx: CommandContext) -> CliResult[JobInfo]:
    """Get status of a background job.

    Parameters
    ----------
    ctx
        Command context with params:
        - job_id: Job ID

    Returns
    -------
    CliResult[JobInfo]
        Job status details.
    """
    job_id = ctx.params.require_str("job_id")

    LOG.info("Getting status for job: %s", job_id)

    job = ctx.jobs.get_status(job_id)

    if job is None:
        return fail_job_not_found(job_id)

    return CliResult.ok(
        JobInfo(
            job_id=job.job_id,
            operation_id=job.operation_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )
    )


def jobs_output_handler(ctx: CommandContext) -> CliResult[JobOutputResult]:
    """Get output of a completed job.

    Parameters
    ----------
    ctx
        Command context with params:
        - job_id: Job ID

    Returns
    -------
    CliResult[JobOutputResult]
        Job output.
    """
    job_id = ctx.params.require_str("job_id")

    LOG.info("Getting output for job: %s", job_id)

    job = ctx.jobs.get_status(job_id)

    if job is None:
        return fail_job_not_found(job_id)

    if job.status != JobStatus.COMPLETED:
        return fail_job_not_completed(job_id, job.status.value)

    output = ctx.jobs.get_output(job_id)

    return CliResult.ok(
        JobOutputResult(
            job_id=job_id,
            has_output=output is not None,
            output=output,
        )
    )


def jobs_cancel_handler(ctx: CommandContext) -> CliResult[ActionResult]:
    """Cancel a running job.

    Parameters
    ----------
    ctx
        Command context with params:
        - job_id: Job ID

    Returns
    -------
    CliResult[ActionResult]
        Cancellation result.
    """
    job_id = ctx.params.require_str("job_id")

    LOG.info("Cancelling job: %s", job_id)

    cancelled = ctx.jobs.cancel(job_id)

    if not cancelled:
        return fail_job_cancel_failed(job_id)

    return CliResult.ok(
        ActionResult(
            action="cancelled",
            success=True,
            affected_count=1,
            message=f"Job {job_id} cancelled",
        )
    )


def jobs_cleanup_handler(ctx: CommandContext) -> CliResult[ActionResult]:
    """Clean up old completed jobs.

    Parameters
    ----------
    ctx
        Command context with params:
        - max_age_days: Maximum age in days (default 7)

    Returns
    -------
    CliResult[ActionResult]
        Cleanup result.
    """
    max_age_days = ctx.params.get_int("max_age_days", 7)

    LOG.info("Cleaning up jobs older than %d days", max_age_days)

    cleaned = ctx.jobs.cleanup(max_age_days=max_age_days)

    return CliResult.ok(
        ActionResult(
            action="cleanup",
            success=True,
            affected_count=cleaned,
            message=f"Cleaned {cleaned} old jobs",
        )
    )


__all__ = [
    "JobInfo",
    "JobOutputResult",
    "jobs_cancel_handler",
    "jobs_cleanup_handler",
    "jobs_list_handler",
    "jobs_output_handler",
    "jobs_status_handler",
]
