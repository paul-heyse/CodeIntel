"""Tests for JobService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.cli.jobs import JobInfo, JobManager, JobStatus, JobStore
from codeintel.cli.services.jobs import JobService


class TestJobServiceCreation:
    """Test JobService creation."""

    def test_default_uses_global_manager(self) -> None:
        """Default service uses global JobManager."""
        service = JobService()
        manager = service.manager
        assert manager is not None

    def test_with_custom_manager(self) -> None:
        """Create with custom manager."""
        custom = MagicMock(spec=JobManager)
        service = JobService(manager=custom)
        assert service.manager is custom

    def test_with_store(self, tmp_path: Path) -> None:
        """Create with custom store."""
        store = JobStore(base_dir=tmp_path)
        service = JobService.with_store(store)
        assert service.manager is not None


class TestJobServiceOperations:
    """Test JobService operations."""

    def test_list_jobs_delegates(self) -> None:
        """List jobs delegates to manager."""
        manager = MagicMock(spec=JobManager)
        expected = [MagicMock(spec=JobInfo)]
        manager.list_jobs.return_value = expected

        service = JobService(manager=manager)
        result = service.list_jobs(status=JobStatus.RUNNING, limit=10)

        assert result == expected
        manager.list_jobs.assert_called_once_with(status=JobStatus.RUNNING, limit=10)

    def test_get_status_delegates(self) -> None:
        """Get status delegates to manager."""
        manager = MagicMock(spec=JobManager)
        expected = MagicMock(spec=JobInfo)
        manager.get_status.return_value = expected

        service = JobService(manager=manager)
        result = service.get_status("job-123")

        assert result == expected
        manager.get_status.assert_called_once_with("job-123")

    def test_get_output_delegates(self) -> None:
        """Get output delegates to manager."""
        manager = MagicMock(spec=JobManager)
        expected = {"result": "data"}
        manager.get_output.return_value = expected

        service = JobService(manager=manager)
        result = service.get_output("job-123")

        assert result == expected
        manager.get_output.assert_called_once_with("job-123")

    def test_submit_delegates(self) -> None:
        """Submit delegates to manager."""
        manager = MagicMock(spec=JobManager)
        manager.submit.return_value = "new-job-id"

        service = JobService(manager=manager)
        result = service.submit("test.operation", {"param": "value"})

        assert result == "new-job-id"
        manager.submit.assert_called_once_with("test.operation", {"param": "value"})

    def test_cancel_delegates(self) -> None:
        """Cancel delegates to manager."""
        manager = MagicMock(spec=JobManager)
        manager.cancel.return_value = True

        service = JobService(manager=manager)
        result = service.cancel("job-123")

        assert result is True
        manager.cancel.assert_called_once_with("job-123")

    def test_cleanup_delegates(self) -> None:
        """Cleanup delegates to manager."""
        manager = MagicMock(spec=JobManager)
        manager.cleanup.return_value = 5

        service = JobService(manager=manager)
        result = service.cleanup(max_age_days=14)

        assert result == 5
        manager.cleanup.assert_called_once_with(max_age_days=14)


class TestJobServiceIntegration:
    """Integration tests for JobService."""

    def test_full_job_lifecycle(self, tmp_path: Path) -> None:
        """Test complete job lifecycle with real store."""
        store = JobStore(base_dir=tmp_path)
        service = JobService.with_store(store)

        # List should be empty initially
        jobs = service.list_jobs()
        assert len(jobs) == 0

        # Note: submit starts a background process, so we don't test it here
        # Just verify the service methods work

        cleanup_count = service.cleanup(max_age_days=30)
        assert cleanup_count == 0  # Nothing to clean
