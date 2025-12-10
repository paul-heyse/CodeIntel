"""Job management operations.

Operations for listing, monitoring, and managing background jobs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from codeintel.cli.jobs import JobStatus, get_job_manager
from codeintel.operations.base import Capability, Operation, operation
from codeintel.operations.context import OpContext
from codeintel.operations.errors.factory import (
    fail_job_cancel_failed,
    fail_job_not_completed,
    fail_job_not_found,
)
from codeintel.operations.result import Result, result_type

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@result_type
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


@result_type
@dataclass(frozen=True)
class JobsListResult:
    """Result from listing jobs."""

    jobs: list[dict[str, object]]
    count: int


@result_type
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


@result_type
@dataclass(frozen=True)
class JobOutputResult:
    """Result from getting job output."""

    job_id: str
    has_output: bool
    output: dict[str, Any] | None = None


@result_type
@dataclass(frozen=True)
class JobCancelResult:
    """Result from cancelling a job."""

    job_id: str
    cancelled: bool


@result_type
@dataclass(frozen=True)
class JobsCleanupResult:
    """Result from cleaning up jobs."""

    cleaned_count: int


# -----------------------------------------------------------------------------
# Parameter Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ListJobsParams:
    """Parameters for listing jobs."""

    status: Literal["pending", "running", "completed", "failed", "cancelled"] | None = None
    limit: int = 20


@dataclass(frozen=True)
class GetJobParams:
    """Parameters for getting job details."""

    job_id: str


@dataclass(frozen=True)
class CleanupJobsParams:
    """Parameters for cleaning up jobs."""

    max_age_days: int = 7


# -----------------------------------------------------------------------------
# Operations
# -----------------------------------------------------------------------------


@operation("jobs.list", capabilities=frozenset({Capability.JOBS_READ}))
class ListJobs(Operation[ListJobsParams, JobsListResult]):
    """List background jobs with optional filtering."""

    __operation_id__: ClassVar[str] = "jobs.list"
    __params_type__: ClassVar[type[ListJobsParams]] = ListJobsParams
    __result_type__: ClassVar[type[JobsListResult]] = JobsListResult
    __capabilities__: ClassVar[frozenset[str]] = frozenset({Capability.JOBS_READ})

    def execute(self, params: ListJobsParams, ctx: OpContext) -> Result[JobsListResult]:
        """Execute the list jobs operation.

        Parameters
        ----------
        params
            List parameters.
        ctx
            Operation context.

        Returns
        -------
        Result[JobsListResult]
            List of jobs.
        """
        _ = (self, ctx)  # Instance method for protocol compatibility

        LOG.info("Listing jobs with status=%s, limit=%d", params.status, params.limit)

        manager = get_job_manager()
        status_filter = JobStatus(params.status) if params.status else None
        jobs = manager.list_jobs(status=status_filter, limit=params.limit)

        job_dicts: list[dict[str, object]] = [j.to_dict() for j in jobs]

        return Result.ok(JobsListResult(jobs=job_dicts, count=len(jobs)))


@operation("jobs.status", capabilities=frozenset({Capability.JOBS_READ}))
class GetJobStatus(Operation[GetJobParams, JobStatusResult]):
    """Get detailed status of a specific job."""

    __operation_id__: ClassVar[str] = "jobs.status"
    __params_type__: ClassVar[type[GetJobParams]] = GetJobParams
    __result_type__: ClassVar[type[JobStatusResult]] = JobStatusResult
    __capabilities__: ClassVar[frozenset[str]] = frozenset({Capability.JOBS_READ})

    def execute(self, params: GetJobParams, ctx: OpContext) -> Result[JobStatusResult]:
        """Execute the get job status operation.

        Parameters
        ----------
        params
            Job ID parameter.
        ctx
            Operation context.

        Returns
        -------
        Result[JobStatusResult]
            Job status details.
        """
        _ = (self, ctx)  # Instance method for protocol compatibility

        LOG.info("Getting status for job: %s", params.job_id)

        manager = get_job_manager()
        job = manager.get_status(params.job_id)

        if job is None:
            return fail_job_not_found(params.job_id)

        return Result.ok(
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


@operation("jobs.output", capabilities=frozenset({Capability.JOBS_READ}))
class GetJobOutput(Operation[GetJobParams, JobOutputResult]):
    """Get output of a completed job."""

    __operation_id__: ClassVar[str] = "jobs.output"
    __params_type__: ClassVar[type[GetJobParams]] = GetJobParams
    __result_type__: ClassVar[type[JobOutputResult]] = JobOutputResult
    __capabilities__: ClassVar[frozenset[str]] = frozenset({Capability.JOBS_READ})

    def execute(self, params: GetJobParams, ctx: OpContext) -> Result[JobOutputResult]:
        """Execute the get job output operation.

        Parameters
        ----------
        params
            Job ID parameter.
        ctx
            Operation context.

        Returns
        -------
        Result[JobOutputResult]
            Job output.
        """
        _ = (self, ctx)  # Instance method for protocol compatibility

        LOG.info("Getting output for job: %s", params.job_id)

        manager = get_job_manager()
        job = manager.get_status(params.job_id)

        if job is None:
            return fail_job_not_found(params.job_id)

        if job.status != JobStatus.COMPLETED:
            return fail_job_not_completed(params.job_id, job.status.value)

        output = manager.get_output(params.job_id)

        return Result.ok(
            JobOutputResult(
                job_id=params.job_id,
                has_output=output is not None,
                output=output,
            )
        )


@operation("jobs.cancel", capabilities=frozenset({Capability.JOBS_WRITE}))
class CancelJob(Operation[GetJobParams, JobCancelResult]):
    """Cancel a running job."""

    __operation_id__: ClassVar[str] = "jobs.cancel"
    __params_type__: ClassVar[type[GetJobParams]] = GetJobParams
    __result_type__: ClassVar[type[JobCancelResult]] = JobCancelResult
    __capabilities__: ClassVar[frozenset[str]] = frozenset({Capability.JOBS_WRITE})

    def execute(self, params: GetJobParams, ctx: OpContext) -> Result[JobCancelResult]:
        """Execute the cancel job operation.

        Parameters
        ----------
        params
            Job ID parameter.
        ctx
            Operation context.

        Returns
        -------
        Result[JobCancelResult]
            Cancellation result.
        """
        _ = (self, ctx)  # Instance method for protocol compatibility

        LOG.info("Cancelling job: %s", params.job_id)

        manager = get_job_manager()
        cancelled = manager.cancel(params.job_id)

        if not cancelled:
            return fail_job_cancel_failed(params.job_id)

        return Result.ok(JobCancelResult(job_id=params.job_id, cancelled=True))


@operation("jobs.cleanup", capabilities=frozenset({Capability.JOBS_WRITE}))
class CleanupJobs(Operation[CleanupJobsParams, JobsCleanupResult]):
    """Clean up old completed jobs."""

    __operation_id__: ClassVar[str] = "jobs.cleanup"
    __params_type__: ClassVar[type[CleanupJobsParams]] = CleanupJobsParams
    __result_type__: ClassVar[type[JobsCleanupResult]] = JobsCleanupResult
    __capabilities__: ClassVar[frozenset[str]] = frozenset({Capability.JOBS_WRITE})

    def execute(self, params: CleanupJobsParams, ctx: OpContext) -> Result[JobsCleanupResult]:
        """Execute the cleanup jobs operation.

        Parameters
        ----------
        params
            Cleanup parameters.
        ctx
            Operation context.

        Returns
        -------
        Result[JobsCleanupResult]
            Cleanup result.
        """
        _ = (self, ctx)  # Instance method for protocol compatibility

        LOG.info("Cleaning up jobs older than %d days", params.max_age_days)

        manager = get_job_manager()
        cleaned = manager.cleanup(max_age_days=params.max_age_days)

        return Result.ok(JobsCleanupResult(cleaned_count=cleaned))


__all__ = [
    "CancelJob",
    "CleanupJobs",
    "CleanupJobsParams",
    "GetJobOutput",
    "GetJobParams",
    "GetJobStatus",
    "JobCancelResult",
    "JobInfo",
    "JobOutputResult",
    "JobStatusResult",
    "JobsCleanupResult",
    "JobsListResult",
    "ListJobs",
    "ListJobsParams",
]
