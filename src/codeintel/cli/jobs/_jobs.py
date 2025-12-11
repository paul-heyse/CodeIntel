"""Background job management.

Provide infrastructure for running operations asynchronously,
tracking their status, and retrieving results.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from multiprocessing import Process
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Information about a background job.

    Parameters
    ----------
    job_id
        Unique job identifier.
    operation_id
        Operation being executed.
    params
        Operation parameters.
    status
        Current status.
    created_at
        Job creation timestamp.
    started_at
        Execution start timestamp.
    completed_at
        Completion timestamp.
    pid
        Process ID if running.
    exit_code
        Exit code if completed.
    error
        Error message if failed.
    """

    job_id: str
    operation_id: str
    params: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobInfo:
        """Create from dictionary.

        Parameters
        ----------
        data
            Dictionary data.

        Returns
        -------
        JobInfo
            Job info instance.
        """
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobStore:
    """Persistent storage for job information.

    Parameters
    ----------
    base_dir
        Base directory for job storage.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize job store."""
        self._base_dir = base_dir or (Path.home() / ".codeintel" / "jobs")
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Get path for job metadata.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        Path
            Metadata file path.
        """
        return self._base_dir / f"{job_id}.json"

    def _output_path(self, job_id: str) -> Path:
        """Get path for job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        Path
            Output file path.
        """
        return self._base_dir / f"{job_id}.output.json"

    def save(self, job: JobInfo) -> None:
        """Save job information.

        Parameters
        ----------
        job
            Job to save.
        """
        path = self._job_path(job.job_id)
        path.write_text(json.dumps(job.to_dict(), indent=2))

    def load(self, job_id: str) -> JobInfo | None:
        """Load job information.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None if not found.
        """
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return JobInfo.from_dict(data)

    def save_output(self, job_id: str, output: dict[str, Any]) -> None:
        """Save job output.

        Parameters
        ----------
        job_id
            Job identifier.
        output
            Output data.
        """
        path = self._output_path(job_id)
        path.write_text(json.dumps(output, indent=2))

    def load_output(self, job_id: str) -> dict[str, Any] | None:
        """Load job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        dict[str, Any] | None
            Output data or None.
        """
        path = self._output_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs.

        Parameters
        ----------
        status
            Filter by status.
        limit
            Maximum jobs to return.

        Returns
        -------
        list[JobInfo]
            Matching jobs.
        """
        jobs = []
        job_files = sorted(self._base_dir.glob("*.json"), reverse=True)

        for path in job_files:
            if path.name.endswith(".output.json"):
                continue
            try:
                job = JobInfo.from_dict(json.loads(path.read_text()))
                if status is None or job.status == status:
                    jobs.append(job)
                if len(jobs) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return jobs

    def delete(self, job_id: str) -> bool:
        """Delete job and its output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        bool
            True if deleted.
        """
        job_path = self._job_path(job_id)
        output_path = self._output_path(job_id)
        deleted = False
        if job_path.exists():
            job_path.unlink()
            deleted = True
        if output_path.exists():
            output_path.unlink()
        return deleted


class JobManager:
    """Manage background job execution.

    Parameters
    ----------
    store
        Job storage backend.
    """

    def __init__(self, store: JobStore | None = None) -> None:
        """Initialize job manager."""
        self._store = store or JobStore()

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
        job_id = str(uuid.uuid4())[:8]

        job = JobInfo(
            job_id=job_id,
            operation_id=operation_id,
            params=params,
            status=JobStatus.PENDING,
        )
        self._store.save(job)

        # Start subprocess
        self._start_job_process(job)

        return job_id

    def _start_job_process(self, job: JobInfo) -> None:
        """Start background process for job.

        Parameters
        ----------
        job
            Job to start.
        """
        process = Process(target=_run_job_process, args=(job.job_id,))
        process.start()

        job.pid = process.pid
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC).isoformat()
        self._store.save(job)

    def get_status(self, job_id: str) -> JobInfo | None:
        """Get job status.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        JobInfo | None
            Job info or None.
        """
        job = self._store.load(job_id)
        if job is None:
            return None
        if (
            job.status == JobStatus.RUNNING
            and job.pid is not None
            and not self._is_process_running(job.pid)
        ):
            job.status = JobStatus.FAILED
            job.error = "Process terminated unexpectedly"
            job.completed_at = datetime.now(UTC).isoformat()
            self._store.save(job)
        return job

    def get_output(self, job_id: str) -> dict[str, Any] | None:
        """Get job output.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        dict[str, Any] | None
            Output data or None.
        """
        return self._store.load_output(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Parameters
        ----------
        job_id
            Job identifier.

        Returns
        -------
        bool
            True if cancelled.
        """
        job = self._store.load(job_id)
        if job is None:
            return False

        if job.status != JobStatus.RUNNING:
            return False

        if job.pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(job.pid, signal.SIGTERM)

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(UTC).isoformat()
        self._store.save(job)
        return True

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobInfo]:
        """List jobs.

        Parameters
        ----------
        status
            Filter by status.
        limit
            Maximum jobs.

        Returns
        -------
        list[JobInfo]
            Jobs.
        """
        return self._store.list_jobs(status=status, limit=limit)

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
        cutoff = datetime.now(UTC).timestamp() - (max_age_days * 86400)
        cleaned = 0

        terminal_statuses = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

        for job in self._store.list_jobs(limit=1000):
            is_terminal = job.status in terminal_statuses
            if not is_terminal or job.completed_at is None:
                continue
            completed_ts = datetime.fromisoformat(job.completed_at).timestamp()
            if completed_ts < cutoff:
                self._store.delete(job.job_id)
                cleaned += 1

        return cleaned

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if process is running.

        Parameters
        ----------
        pid
            Process ID.

        Returns
        -------
        bool
            True if running.
        """
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True


def run_job(job_id: str) -> int:
    """Execute a queued job inside the current process.

    Parameters
    ----------
    job_id
        Job identifier.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on failure).
    """
    store = JobStore()
    job = store.load(job_id)

    if job is None:
        return 1
    # Import lazily to avoid circular imports during CLI startup.
    from codeintel.cli.execution.registry import (  # noqa: PLC0415
        execute_operation,
        get_registry,
    )

    registry = get_registry()
    spec = registry.get(job.operation_id)

    if spec is None:
        job.status = JobStatus.FAILED
        job.error = f"Unknown operation: {job.operation_id}"
        job.completed_at = datetime.now(UTC).isoformat()
        store.save(job)
        return 1

    try:
        result = execute_operation(spec, job.params)

        if result.success:
            job.status = JobStatus.COMPLETED
            store.save_output(job.job_id, result.to_dict())
        else:
            job.status = JobStatus.FAILED
            error_detail = ""
            if result.error:
                error_detail = result.error.detail or "Unknown error"
            job.error = error_detail

        job.exit_code = 0 if result.success else 1

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.exit_code = 1

    job.completed_at = datetime.now(UTC).isoformat()
    store.save(job)

    return job.exit_code or 0


def _run_job_process(job_id: str) -> None:
    """Run a job in a child process."""
    exit_code = run_job(job_id)
    sys.exit(exit_code)


# Global job manager
_JOB_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get global job manager.

    Returns
    -------
    JobManager
        Job manager instance.
    """
    global _JOB_MANAGER  # noqa: PLW0603
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER


__all__ = [
    "JobInfo",
    "JobManager",
    "JobStatus",
    "JobStore",
    "get_job_manager",
    "run_job",
]
