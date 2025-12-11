"""Background job management handlers.

Handlers for job listing, status, and management operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_job_cancel_failed,
    fail_job_not_completed,
    fail_job_not_found,
)
from codeintel.cli.jobs import JobStatus, get_job_manager

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobInfo:
    """Information about a single job."""

    job_id: str
    operation_id: str
    status: str
    created_at: str | None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "job_id": self.job_id,
            "operation_id": self.operation_id,
            "status": self.status,
            "created_at": self.created_at,
        }
        if self.started_at:
            result["started_at"] = self.started_at
        if self.completed_at:
            result["completed_at"] = self.completed_at
        if self.error:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class JobsListResult:
    """Result from listing jobs."""

    jobs: list[dict[str, object]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "jobs": self.jobs,
            "count": self.count,
        }


@dataclass(frozen=True)
class JobStatusResult:
    """Result from getting job status."""

    job_id: str
    operation_id: str
    status: str
    created_at: str | None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "job_id": self.job_id,
            "operation_id": self.operation_id,
            "status": self.status,
            "created_at": self.created_at,
        }
        if self.started_at:
            result["started_at"] = self.started_at
        if self.completed_at:
            result["completed_at"] = self.completed_at
        if self.error:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class JobOutputResult:
    """Result from getting job output."""

    job_id: str
    has_output: bool
    output: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "job_id": self.job_id,
            "has_output": self.has_output,
        }
        if self.output:
            result["output"] = self.output
        return result


@dataclass(frozen=True)
class JobCancelResult:
    """Result from cancelling a job."""

    job_id: str
    cancelled: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "job_id": self.job_id,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True)
class JobsCleanupResult:
    """Result from cleaning up jobs."""

    cleaned_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "cleaned_count": self.cleaned_count,
        }


def jobs_list_handler(ctx: CommandContext) -> CliResult[JobsListResult]:
    """List background jobs.

    Parameters
    ----------
    ctx
        Command context with params:
        - status: Optional status filter
        - limit: Maximum jobs to return (default 20)

    Returns
    -------
    CliResult[JobsListResult]
        List of jobs.
    """
    status_str = ctx.params.get_str("status")
    limit = ctx.params.get_int("limit", 20)

    LOG.info("Listing jobs with status=%s, limit=%d", status_str, limit)

    manager = get_job_manager()
    status_filter = JobStatus(status_str) if status_str else None
    jobs = manager.list_jobs(status=status_filter, limit=limit)

    job_dicts: list[dict[str, object]] = [j.to_dict() for j in jobs]

    return CliResult.ok(JobsListResult(jobs=job_dicts, count=len(jobs)))


def jobs_status_handler(ctx: CommandContext) -> CliResult[JobStatusResult]:
    """Get status of a background job.

    Parameters
    ----------
    ctx
        Command context with params:
        - job_id: Job ID

    Returns
    -------
    CliResult[JobStatusResult]
        Job status details.
    """
    job_id = ctx.params.require_str("job_id")

    LOG.info("Getting status for job: %s", job_id)

    manager = get_job_manager()
    job = manager.get_status(job_id)

    if job is None:
        return fail_job_not_found(job_id)

    return CliResult.ok(
        JobStatusResult(
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

    manager = get_job_manager()
    job = manager.get_status(job_id)

    if job is None:
        return fail_job_not_found(job_id)

    if job.status != JobStatus.COMPLETED:
        return fail_job_not_completed(job_id, job.status.value)

    output = manager.get_output(job_id)

    return CliResult.ok(
        JobOutputResult(
            job_id=job_id,
            has_output=output is not None,
            output=output,
        )
    )


def jobs_cancel_handler(ctx: CommandContext) -> CliResult[JobCancelResult]:
    """Cancel a running job.

    Parameters
    ----------
    ctx
        Command context with params:
        - job_id: Job ID

    Returns
    -------
    CliResult[JobCancelResult]
        Cancellation result.
    """
    job_id = ctx.params.require_str("job_id")

    LOG.info("Cancelling job: %s", job_id)

    manager = get_job_manager()
    cancelled = manager.cancel(job_id)

    if not cancelled:
        return fail_job_cancel_failed(job_id)

    return CliResult.ok(JobCancelResult(job_id=job_id, cancelled=True))


def jobs_cleanup_handler(ctx: CommandContext) -> CliResult[JobsCleanupResult]:
    """Clean up old completed jobs.

    Parameters
    ----------
    ctx
        Command context with params:
        - max_age_days: Maximum age in days (default 7)

    Returns
    -------
    CliResult[JobsCleanupResult]
        Cleanup result.
    """
    max_age_days = ctx.params.get_int("max_age_days", 7)

    LOG.info("Cleaning up jobs older than %d days", max_age_days)

    manager = get_job_manager()
    cleaned = manager.cleanup(max_age_days=max_age_days)

    return CliResult.ok(JobsCleanupResult(cleaned_count=cleaned))


__all__ = [
    "JobCancelResult",
    "JobInfo",
    "JobOutputResult",
    "JobStatusResult",
    "JobsCleanupResult",
    "JobsListResult",
    "jobs_cancel_handler",
    "jobs_cleanup_handler",
    "jobs_list_handler",
    "jobs_output_handler",
    "jobs_status_handler",
]
