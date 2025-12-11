"""Tests for JobService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.cli.jobs import JobInfo, JobManager, JobStatus, JobStore
from codeintel.cli.services.jobs import JobService
from tests._helpers.assertions import expect_equal, expect_is_not_none, expect_true

# ---------------------------------------------------------------------------
# JobService creation
# ---------------------------------------------------------------------------


def test_default_uses_global_manager() -> None:
    """Default service uses global JobManager."""
    service = JobService()
    manager = service.manager
    expect_is_not_none(manager)


def test_with_custom_manager() -> None:
    """Create with custom manager."""
    custom = MagicMock(spec=JobManager)
    service = JobService(manager=custom)
    expect_true(service.manager is custom)


def test_with_store(tmp_path: Path) -> None:
    """Create with custom store."""
    store = JobStore(base_dir=tmp_path)
    service = JobService.with_store(store)
    expect_is_not_none(service.manager)


# ---------------------------------------------------------------------------
# JobService operations
# ---------------------------------------------------------------------------


def test_list_jobs_delegates() -> None:
    """List jobs delegates to manager."""
    manager = MagicMock(spec=JobManager)
    expected = [MagicMock(spec=JobInfo)]
    manager.list_jobs.return_value = expected

    service = JobService(manager=manager)
    result = service.list_jobs(status=JobStatus.RUNNING, limit=10)

    expect_equal(result, expected)
    manager.list_jobs.assert_called_once_with(status=JobStatus.RUNNING, limit=10)


def test_get_status_delegates() -> None:
    """Get status delegates to manager."""
    manager = MagicMock(spec=JobManager)
    expected = MagicMock(spec=JobInfo)
    manager.get_status.return_value = expected

    service = JobService(manager=manager)
    result = service.get_status("job-123")

    expect_equal(result, expected)
    manager.get_status.assert_called_once_with("job-123")


def test_get_output_delegates() -> None:
    """Get output delegates to manager."""
    manager = MagicMock(spec=JobManager)
    expected = {"result": "data"}
    manager.get_output.return_value = expected

    service = JobService(manager=manager)
    result = service.get_output("job-123")

    expect_equal(result, expected)
    manager.get_output.assert_called_once_with("job-123")


def test_submit_delegates() -> None:
    """Submit delegates to manager."""
    manager = MagicMock(spec=JobManager)
    manager.submit.return_value = "new-job-id"

    service = JobService(manager=manager)
    result = service.submit("test.operation", {"param": "value"})

    expect_equal(result, "new-job-id")
    manager.submit.assert_called_once_with("test.operation", {"param": "value"})


def test_cancel_delegates() -> None:
    """Cancel delegates to manager."""
    manager = MagicMock(spec=JobManager)
    manager.cancel.return_value = True

    service = JobService(manager=manager)
    result = service.cancel("job-123")

    expect_true(result)
    manager.cancel.assert_called_once_with("job-123")


def test_cleanup_delegates() -> None:
    """Cleanup delegates to manager."""
    manager = MagicMock(spec=JobManager)
    manager.cleanup.return_value = 5

    service = JobService(manager=manager)
    result = service.cleanup(max_age_days=14)

    expect_equal(result, 5)
    manager.cleanup.assert_called_once_with(max_age_days=14)


# ---------------------------------------------------------------------------
# JobService integration
# ---------------------------------------------------------------------------


def test_full_job_lifecycle(tmp_path: Path) -> None:
    """Test complete job lifecycle with real store."""
    store = JobStore(base_dir=tmp_path)
    service = JobService.with_store(store)

    # List should be empty initially
    jobs = service.list_jobs()
    expect_equal(len(jobs), 0)

    # Note: submit starts a background process, so we don't test it here
    # Just verify the service methods work

    cleanup_count = service.cleanup(max_age_days=30)
    expect_equal(cleanup_count, 0)
