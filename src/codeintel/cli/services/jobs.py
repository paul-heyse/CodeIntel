"""Background job management service.

Provide a service wrapper around the JobManager for consistent API access
through the CommandContext.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.cli.deps.protocols import JobManagerProtocol
from codeintel.cli.jobs import JobInfo, JobManager, JobStatus, get_job_manager

if TYPE_CHECKING:
    from codeintel.cli.jobs import JobStore


class JobService:
    """Background job management service.

    Wrap JobManager to provide consistent API through CommandContext.
    Support both global singleton and custom manager instances.

    Parameters
    ----------
    manager
        Optional custom JobManager. If None, uses global singleton.

    Examples
    --------
    >>> service = JobService()
    >>> jobs = service.list_jobs(limit=10)
    >>> job = service.get_status("abc123")
    """

    def __init__(self, manager: JobManagerProtocol | None = None) -> None:
        """Initialize job service."""
        self._manager = manager

    @property
    def manager(self) -> JobManagerProtocol:
        """Get the job manager instance.

        Returns
        -------
        JobManagerProtocol
            Job manager (singleton or provided instance).
        """
        if self._manager is None:
            return get_job_manager()
        return self._manager

    @classmethod
    def with_store(cls, store: JobStore) -> JobService:
        """Create with custom job store.

        Parameters
        ----------
        store
            Custom job storage backend.

        Returns
        -------
        JobService
            Service with custom store.
        """
        return cls(manager=JobManager(store=store))

    def submit(
        self,
        operation_id: str,
        params: dict[str, Any],
    ) -> str:
        """Submit a job for background execution.

        Parameters
        ----------
        operation_id
            Operation to execute.
        params
            Operation parameters.

        Returns
        -------
        str
            Job ID.
        """
        return self.manager.submit(operation_id, params)

    def get_status(self, job_id: str) -> JobInfo | None:
        """Get job status.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None if not found.
        """
        return self.manager.get_status(job_id)

    def get_output(self, job_id: str) -> dict[str, Any] | None:
        """Get output from completed job.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        dict[str, Any] | None
            Output data or None.
        """
        return self.manager.get_output(job_id)

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs with optional filters.

        Parameters
        ----------
        status
            Filter by job status.
        limit
            Maximum jobs to return.

        Returns
        -------
        list[JobInfo]
            Matching jobs.
        """
        return self.manager.list_jobs(status=status, limit=limit)

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        bool
            True if cancelled successfully.
        """
        return self.manager.cancel(job_id)

    def cleanup(self, *, max_age_days: int = 7) -> int:
        """Clean up old completed jobs.

        Parameters
        ----------
        max_age_days
            Maximum age in days.

        Returns
        -------
        int
            Number of jobs cleaned.
        """
        return self.manager.cleanup(max_age_days=max_age_days)


__all__ = [
    "JobService",
]
