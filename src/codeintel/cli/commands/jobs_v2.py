"""Background job management commands (new Command[T] pattern).

This module demonstrates the new Command[T] pattern with explicit
dependencies via Deps container.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.result_types import ActionResult, ListResult
from codeintel.cli.core.results import result_type
from codeintel.cli.errors.factory import (
    fail_job_cancel_failed,
    fail_job_not_completed,
    fail_job_not_found,
)
from codeintel.cli.jobs import JobStatus

if TYPE_CHECKING:
    from codeintel.cli.deps import Deps

LOG = logging.getLogger(__name__)

jobs_v2_app = App(name="jobs", help="Manage background jobs (v2)")


# =============================================================================
# Result Types
# =============================================================================


@result_type
@dataclass(frozen=True)
class JobInfo:
    """Information about a single job.

    Parameters
    ----------
    job_id
        Unique job identifier.
    operation_id
        Operation that was executed.
    status
        Current job status.
    created_at
        Creation timestamp.
    started_at
        Start timestamp.
    completed_at
        Completion timestamp.
    error
        Error message if failed.
    """

    job_id: str
    operation_id: str
    status: str
    created_at: str | None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@result_type
@dataclass(frozen=True)
class JobOutputResult:
    """Result from getting job output.

    Parameters
    ----------
    job_id
        Job identifier.
    has_output
        Whether output is available.
    output
        Output data if available.
    """

    job_id: str
    has_output: bool
    output: dict[str, Any] | None = None


# =============================================================================
# Commands
# =============================================================================


@cli_command("jobs_v2.list", require_storage=False)
@jobs_v2_app.command(name="list")
@dataclass(frozen=True)
class ListJobs(Command[ListResult[JobInfo]]):
    """List background jobs.

    Display a table of all background jobs with their status,
    operation, and timestamps.
    """

    __operation_id__ = "jobs_v2.list"

    status: Annotated[
        Literal["pending", "running", "completed", "failed", "cancelled"] | None,
        Parameter(help="Filter by status"),
    ] = None
    limit: Annotated[int, Parameter(help="Maximum jobs to show")] = 20
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[ListResult[JobInfo]]:
        """Execute job listing.

        Parameters
        ----------
        deps
            Dependencies container with job manager.

        Returns
        -------
        CliResult[ListResult[JobInfo]]
            List of jobs.
        """
        LOG.info("Listing jobs with status=%s, limit=%d", self.status, self.limit)

        status_filter = JobStatus(self.status) if self.status else None
        jobs = deps.jobs.list_jobs(status=status_filter, limit=self.limit)

        items = [
            JobInfo(
                job_id=j.job_id,
                operation_id=j.operation_id,
                status=j.status.value,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                error=j.error,
            )
            for j in jobs
        ]

        return CliResult.ok(ListResult.from_items(items))


@cli_command("jobs_v2.status", require_storage=False)
@jobs_v2_app.command(name="status")
@dataclass(frozen=True)
class GetJobStatus(Command[JobInfo]):
    """Get status of a background job.

    Display detailed status information for a specific job
    including timestamps and error messages.
    """

    __operation_id__ = "jobs_v2.status"

    job_id: Annotated[str, Parameter(help="Job ID")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[JobInfo]:
        """Execute job status query.

        Parameters
        ----------
        deps
            Dependencies container with job manager.

        Returns
        -------
        CliResult[JobInfo]
            Job status details.
        """
        LOG.info("Getting status for job: %s", self.job_id)

        job = deps.jobs.get_status(self.job_id)

        if job is None:
            return fail_job_not_found(self.job_id)

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


@cli_command("jobs_v2.output", require_storage=False)
@jobs_v2_app.command(name="output")
@dataclass(frozen=True)
class GetJobOutput(Command[JobOutputResult]):
    """Get output of a completed job.

    Retrieve and display the result data from a completed
    background job.
    """

    __operation_id__ = "jobs_v2.output"

    job_id: Annotated[str, Parameter(help="Job ID")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[JobOutputResult]:
        """Execute job output retrieval.

        Parameters
        ----------
        deps
            Dependencies container with job manager.

        Returns
        -------
        CliResult[JobOutputResult]
            Job output.
        """
        LOG.info("Getting output for job: %s", self.job_id)

        job = deps.jobs.get_status(self.job_id)

        if job is None:
            return fail_job_not_found(self.job_id)

        if job.status != JobStatus.COMPLETED:
            return fail_job_not_completed(self.job_id, job.status.value)

        output = deps.jobs.get_output(self.job_id)

        return CliResult.ok(
            JobOutputResult(
                job_id=self.job_id,
                has_output=output is not None,
                output=output,
            )
        )


@cli_command("jobs_v2.cancel", require_storage=False)
@jobs_v2_app.command(name="cancel")
@dataclass(frozen=True)
class CancelJob(Command[ActionResult]):
    """Cancel a running job.

    Send a termination signal to a running job and mark
    it as cancelled.
    """

    __operation_id__ = "jobs_v2.cancel"

    job_id: Annotated[str, Parameter(help="Job ID")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[ActionResult]:
        """Execute job cancellation.

        Parameters
        ----------
        deps
            Dependencies container with job manager.

        Returns
        -------
        CliResult[ActionResult]
            Cancellation result.
        """
        LOG.info("Cancelling job: %s", self.job_id)

        cancelled = deps.jobs.cancel(self.job_id)

        if not cancelled:
            return fail_job_cancel_failed(self.job_id)

        return CliResult.ok(
            ActionResult(
                action="cancelled",
                success=True,
                affected_count=1,
                message=f"Job {self.job_id} cancelled",
            )
        )


@cli_command("jobs_v2.cleanup", require_storage=False)
@jobs_v2_app.command(name="cleanup")
@dataclass(frozen=True)
class CleanupJobs(Command[ActionResult]):
    """Clean up old completed jobs.

    Remove job metadata and output files for jobs that
    completed more than the specified number of days ago.
    """

    __operation_id__ = "jobs_v2.cleanup"

    max_age_days: Annotated[int, Parameter(help="Maximum age in days")] = 7
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)

    def execute(self, deps: Deps) -> CliResult[ActionResult]:
        """Execute job cleanup.

        Parameters
        ----------
        deps
            Dependencies container with job manager.

        Returns
        -------
        CliResult[ActionResult]
            Cleanup result.
        """
        LOG.info("Cleaning up jobs older than %d days", self.max_age_days)

        cleaned = deps.jobs.cleanup(max_age_days=self.max_age_days)

        return CliResult.ok(
            ActionResult(
                action="cleanup",
                success=True,
                affected_count=cleaned,
                message=f"Cleaned {cleaned} old jobs",
            )
        )


__all__ = [
    "CancelJob",
    "CleanupJobs",
    "GetJobOutput",
    "GetJobStatus",
    "JobInfo",
    "JobOutputResult",
    "ListJobs",
    "jobs_v2_app",
]
