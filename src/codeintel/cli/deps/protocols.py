"""Protocols for injectable dependencies.

Each protocol defines a focused interface. Commands declare which dependencies
they need, and the execution infrastructure provides appropriate implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.cli.jobs import JobInfo, JobStatus
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class StorageAccess(Protocol):
    """Protocol for storage layer access.

    Provides read-only and write-enabled gateway access with proper
    lifecycle management.
    """

    @property
    def gateway(self) -> StorageGateway:
        """Get the storage gateway (lazy-loaded, read-only).

        Returns
        -------
        StorageGateway
            Open storage gateway for read operations.
        """
        ...

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway.

        Yields
        ------
        StorageGateway
            Write-enabled storage gateway that will be closed on exit.
        """
        ...


@runtime_checkable
class JobManagerProtocol(Protocol):
    """Protocol for background job management.

    Defines the interface for listing, querying, and managing background jobs.
    """

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
        ...

    def get_status(self, job_id: str) -> JobInfo | None:
        """Get a specific job's status.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None if not found.
        """
        ...

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
        ...

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
        ...

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
        ...

    def submit(self, operation_id: str, params: dict[str, Any]) -> str:
        """Submit a background job."""
        ...


@runtime_checkable
class ServingAccess(Protocol):
    """Protocol for serving layer operations.

    Provides access to the serving operation catalog and invocation.
    """

    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
        *,
        skip_prereqs: bool = False,
    ) -> dict[str, object]:
        """Invoke a serving operation.

        Parameters
        ----------
        operation_id
            Operation ID in the serving catalog.
        params
            Operation parameters.
        skip_prereqs
            If True, skip prerequisite pipeline execution.

        Returns
        -------
        dict[str, object]
            Operation result.

        Raises
        ------
        ValueError
            If operation not found.
        """
        ...


__all__ = [
    "JobManagerProtocol",
    "ServingAccess",
    "StorageAccess",
]
