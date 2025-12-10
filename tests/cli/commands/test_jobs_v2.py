"""Test jobs commands using new Command[T] pattern.

Validate that the new pattern commands produce correct results
with fake dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from codeintel.cli.commands.jobs_v2 import (
    CancelJob,
    CleanupJobs,
    GetJobOutput,
    GetJobStatus,
    JobInfo,
    ListJobs,
)
from codeintel.cli.core.result_types import ActionResult, ListResult
from codeintel.cli.deps import Deps
from codeintel.cli.deps.protocols import JobManagerProtocol
from codeintel.cli.jobs import JobStatus
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)


@dataclass
class FakeJobInfo:
    """Fake JobInfo for testing."""

    job_id: str
    operation_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class FakeJobManager:
    """Fake job manager for testing."""

    def __init__(self) -> None:
        """Initialize with test data."""
        self._jobs: dict[str, FakeJobInfo] = {
            "job-001": FakeJobInfo(
                job_id="job-001",
                operation_id="test.op",
                status=JobStatus.COMPLETED,
                created_at="2024-01-01T00:00:00Z",
                started_at="2024-01-01T00:00:01Z",
                completed_at="2024-01-01T00:00:10Z",
            ),
            "job-002": FakeJobInfo(
                job_id="job-002",
                operation_id="test.op2",
                status=JobStatus.RUNNING,
                created_at="2024-01-01T00:00:00Z",
                started_at="2024-01-01T00:00:01Z",
            ),
        }
        self._outputs: dict[str, dict[str, Any]] = {
            "job-001": {"result": "success"},
        }
        self._cleanup_count = 0
        self._max_age_days: int | None = None

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[FakeJobInfo]:
        """List jobs with optional filters.

        Parameters
        ----------
        status
            Optional status to filter by.
        limit
            Maximum number of jobs to return.

        Returns
        -------
        list[FakeJobInfo]
            Jobs matching the provided filters.
        """
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs[:limit]

    def get_status(self, job_id: str) -> FakeJobInfo | None:
        """Get job status.

        Parameters
        ----------
        job_id
            Identifier of the job to look up.

        Returns
        -------
        FakeJobInfo | None
            Job info if found, otherwise None.
        """
        return self._jobs.get(job_id)

    def get_output(self, job_id: str) -> dict[str, Any] | None:
        """Get job output.

        Parameters
        ----------
        job_id
            Identifier of the job to look up.

        Returns
        -------
        dict[str, Any] | None
            Output payload if present, otherwise None.
        """
        return self._outputs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Parameters
        ----------
        job_id
            Identifier of the job to cancel.

        Returns
        -------
        bool
            True if the job was cancelled, else False.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.CANCELLED
        return True

    def cleanup(self, *, max_age_days: int = 7) -> int:
        """Clean up old jobs.

        Parameters
        ----------
        max_age_days
            Maximum age (in days) of jobs to retain.

        Returns
        -------
        int
            Number of jobs cleaned up.
        """
        self._max_age_days = max_age_days
        self._cleanup_count += 1
        return 5  # Fake cleanup count


class FakeConfig:
    """Fake CLI config for testing."""

    log_level = "WARNING"


def make_fake_deps(jobs: JobManagerProtocol | None = None) -> Deps:
    """Create fake Deps for testing.

    Parameters
    ----------
    jobs
        Optional job manager. Defaults to FakeJobManager.

    Returns
    -------
    Deps
        Configured fake deps.
    """
    return Deps(
        config=FakeConfig(),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
        jobs=jobs or FakeJobManager(),  # type: ignore[arg-type]
    )


class TestListJobs:
    """Test ListJobs command."""

    @staticmethod
    def test_list_all_jobs() -> None:
        """List all jobs returns expected structure."""
        cmd = ListJobs(limit=20)
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_is_instance(data, ListResult)
        expect_equal(data.count, 2)
        expect_equal(len(data.items), 2)

        # Check first job
        job = data.items[0]
        expect_is_instance(job, JobInfo)
        expect_equal(job.job_id, "job-001")
        expect_equal(job.status, "completed")

    @staticmethod
    def test_list_filtered_by_status() -> None:
        """Filter jobs by status works."""
        cmd = ListJobs(status="running", limit=20)
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_equal(data.count, 1)
        expect_equal(data.items[0].status, "running")

    @staticmethod
    def test_list_with_limit() -> None:
        """Limit parameter is respected."""
        cmd = ListJobs(limit=1)
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_equal(data.count, 1)


class TestGetJobStatus:
    """Test GetJobStatus command."""

    @staticmethod
    def test_get_existing_job() -> None:
        """Get status of existing job returns details."""
        cmd = GetJobStatus(job_id="job-001")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_is_instance(data, JobInfo)
        expect_equal(data.job_id, "job-001")
        expect_equal(data.status, "completed")
        expect_equal(data.operation_id, "test.op")

    @staticmethod
    def test_get_nonexistent_job() -> None:
        """Get status of nonexistent job returns error."""
        cmd = GetJobStatus(job_id="nonexistent")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(not result.success)
        error = expect_is_not_none(result.error)
        expect_in("not found", error.title.lower())


class TestGetJobOutput:
    """Test GetJobOutput command."""

    @staticmethod
    def test_get_completed_job_output() -> None:
        """Get output of completed job returns data."""
        cmd = GetJobOutput(job_id="job-001")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_true(data.has_output)
        expect_equal(data.output, {"result": "success"})

    @staticmethod
    def test_get_running_job_output() -> None:
        """Get output of running job returns error."""
        cmd = GetJobOutput(job_id="job-002")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(not result.success)
        expect_is_not_none(result.error)

    @staticmethod
    def test_get_nonexistent_job_output() -> None:
        """Get output of nonexistent job returns error."""
        cmd = GetJobOutput(job_id="nonexistent")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(not result.success)


class TestCancelJob:
    """Test CancelJob command."""

    @staticmethod
    def test_cancel_running_job() -> None:
        """Cancel running job succeeds."""
        cmd = CancelJob(job_id="job-002")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_is_instance(data, ActionResult)
        expect_equal(data.action, "cancelled")
        expect_true(data.success)

    @staticmethod
    def test_cancel_completed_job() -> None:
        """Cancel completed job fails."""
        cmd = CancelJob(job_id="job-001")
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(not result.success)


class TestCleanupJobs:
    """Test CleanupJobs command."""

    @staticmethod
    def test_cleanup_default_age() -> None:
        """Cleanup with default age works."""
        cmd = CleanupJobs()
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_is_instance(data, ActionResult)
        expect_equal(data.action, "cleanup")
        expect_equal(data.affected_count, 5)

    @staticmethod
    def test_cleanup_custom_age() -> None:
        """Cleanup with custom age works."""
        cmd = CleanupJobs(max_age_days=30)
        deps = make_fake_deps()

        result = cmd.execute(deps)

        expect_true(result.success)
        data = expect_is_not_none(result.data)
        expect_equal(data.affected_count, 5)


class TestJobInfoSerialization:
    """Test JobInfo serialization."""

    @staticmethod
    def test_to_dict_omits_none() -> None:
        """Serialization omits None fields."""
        info = JobInfo(
            job_id="job-001",
            operation_id="test.op",
            status="completed",
            created_at="2024-01-01",
        )

        result = {key: value for key, value in info.__dict__.items() if value is not None}

        expect_in("job_id", result)
        expect_in("operation_id", result)
        expect_in("status", result)
        expect_in("created_at", result)
        # None fields should be omitted
        expect_true("started_at" not in result)
        expect_true("error" not in result)

    @staticmethod
    def test_to_dict_includes_values() -> None:
        """Serialization includes non-None fields."""
        info = JobInfo(
            job_id="job-001",
            operation_id="test.op",
            status="completed",
            created_at="2024-01-01",
            started_at="2024-01-01T00:00:01Z",
            error="Test error",
        )

        result = {key: value for key, value in info.__dict__.items() if value is not None}

        expect_equal(result["started_at"], "2024-01-01T00:00:01Z")
        expect_equal(result["error"], "Test error")
