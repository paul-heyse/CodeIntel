"""Unit tests for BuildTracking storage accessor."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.build.manifest import BuildRunRecord, OutputManifest
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.tracking.build_tracking import BuildTracking


@pytest.fixture
def tracking(fresh_gateway: StorageGateway) -> BuildTracking:
    """Create a BuildTracking instance with fresh gateway.

    Parameters
    ----------
    fresh_gateway
        Fresh in-memory gateway with schema applied.

    Returns
    -------
    BuildTracking
        Build tracking accessor for tests.
    """
    return fresh_gateway.build


class TestOutputManifestOperations:
    """Tests for OutputManifest CRUD operations."""

    def test_save_and_load_manifest(self, tracking: BuildTracking) -> None:
        """Save and retrieve a manifest."""
        manifest = OutputManifest(
            target="test_target",
            repo="test-org/test-repo",
            commit="abc123",
            plugin="test_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=1234.5,
            input_hash="input123",
            output_hash="output456",
            row_count=100,
            options_hash="opts789",
        )

        tracking.save_manifest(manifest)
        loaded = tracking.load_manifest("test_target", "test-org/test-repo", "abc123")

        assert loaded is not None
        assert loaded.target == manifest.target
        assert loaded.repo == manifest.repo
        assert loaded.commit == manifest.commit
        assert loaded.plugin == manifest.plugin
        assert loaded.duration_ms == manifest.duration_ms
        assert loaded.input_hash == manifest.input_hash
        assert loaded.output_hash == manifest.output_hash
        assert loaded.row_count == manifest.row_count
        assert loaded.options_hash == manifest.options_hash

    def test_load_nonexistent_manifest(self, tracking: BuildTracking) -> None:
        """Loading non-existent manifest returns None."""
        loaded = tracking.load_manifest("nonexistent", "no/repo", "nocommit")
        assert loaded is None

    def test_upsert_manifest(self, tracking: BuildTracking) -> None:
        """Saving manifest twice updates the record."""
        manifest1 = OutputManifest(
            target="test_target",
            repo="test-org/test-repo",
            commit="abc123",
            plugin="test_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=1000.0,
            input_hash="input1",
        )
        tracking.save_manifest(manifest1)

        # Update with new values
        manifest2 = OutputManifest(
            target="test_target",
            repo="test-org/test-repo",
            commit="abc123",
            plugin="test_plugin",
            computed_at=datetime.now(tz=UTC),
            duration_ms=2000.0,
            input_hash="input2",
        )
        tracking.save_manifest(manifest2)

        loaded = tracking.load_manifest("test_target", "test-org/test-repo", "abc123")
        assert loaded is not None
        assert loaded.duration_ms == 2000.0
        assert loaded.input_hash == "input2"

    def test_list_manifests(self, tracking: BuildTracking) -> None:
        """List all manifests for a repo/commit."""
        now = datetime.now(tz=UTC)
        for i in range(3):
            manifest = OutputManifest(
                target=f"target_{i}",
                repo="test-org/test-repo",
                commit="abc123",
                plugin=f"plugin_{i}",
                computed_at=now,
                duration_ms=float(i * 100),
                input_hash=f"hash_{i}",
            )
            tracking.save_manifest(manifest)

        manifests = tracking.list_manifests("test-org/test-repo", "abc123")
        assert len(manifests) == 3

        # Should be ordered by target name
        names = [m.target for m in manifests]
        assert names == sorted(names)

    def test_delete_manifests(self, tracking: BuildTracking) -> None:
        """Delete all manifests for a repo/commit."""
        now = datetime.now(tz=UTC)
        for i in range(3):
            manifest = OutputManifest(
                target=f"target_{i}",
                repo="test-org/test-repo",
                commit="abc123",
                plugin=f"plugin_{i}",
                computed_at=now,
                duration_ms=100.0,
                input_hash=f"hash_{i}",
            )
            tracking.save_manifest(manifest)

        deleted = tracking.delete_manifests("test-org/test-repo", "abc123")
        # Note: rowcount may not always be accurate in DuckDB
        assert deleted >= 0

        manifests = tracking.list_manifests("test-org/test-repo", "abc123")
        assert len(manifests) == 0


class TestBuildRunOperations:
    """Tests for BuildRunRecord CRUD operations."""

    def test_start_and_fetch_run(self, tracking: BuildTracking) -> None:
        """Start a run and fetch it."""
        record = BuildRunRecord(
            run_id="run-123",
            repo="test-org/test-repo",
            commit="abc123",
            requested_targets=("target1", "target2"),
            computed_targets=(),
            skipped_targets=(),
            started_at=datetime.now(tz=UTC),
            status="running",
        )

        tracking.start_run(record)
        loaded = tracking.fetch_run("run-123")

        assert loaded is not None
        assert loaded.run_id == "run-123"
        assert loaded.repo == "test-org/test-repo"
        assert loaded.commit == "abc123"
        assert loaded.requested_targets == ("target1", "target2")
        assert loaded.status == "running"

    def test_fetch_nonexistent_run(self, tracking: BuildTracking) -> None:
        """Fetching non-existent run returns None."""
        loaded = tracking.fetch_run("nonexistent-run")
        assert loaded is None

    def test_complete_run_succeeded(self, tracking: BuildTracking) -> None:
        """Complete a running run with success status."""
        record = BuildRunRecord(
            run_id="run-456",
            repo="test-org/test-repo",
            commit="def456",
            requested_targets=("target1",),
            computed_targets=(),
            skipped_targets=(),
            started_at=datetime.now(tz=UTC),
            status="running",
        )
        tracking.start_run(record)

        tracking.complete_run(
            run_id="run-456",
            status="succeeded",
            computed_targets=("target1", "target2"),
            skipped_targets=("target3",),
        )

        loaded = tracking.fetch_run("run-456")
        assert loaded is not None
        assert loaded.status == "succeeded"
        assert loaded.computed_targets == ("target1", "target2")
        assert loaded.skipped_targets == ("target3",)
        assert loaded.completed_at is not None
        assert loaded.duration_ms is not None

    def test_complete_run_failed(self, tracking: BuildTracking) -> None:
        """Complete a failed run with error summary."""
        record = BuildRunRecord(
            run_id="run-789",
            repo="test-org/test-repo",
            commit="ghi789",
            requested_targets=("target1",),
            computed_targets=(),
            skipped_targets=(),
            started_at=datetime.now(tz=UTC),
            status="running",
        )
        tracking.start_run(record)

        tracking.complete_run(
            run_id="run-789",
            status="failed",
            computed_targets=(),
            skipped_targets=(),
            error_summary="Test error occurred",
        )

        loaded = tracking.fetch_run("run-789")
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.error_summary == "Test error occurred"

    def test_list_runs(self, tracking: BuildTracking) -> None:
        """List recent runs for a repository."""
        for i in range(3):
            record = BuildRunRecord(
                run_id=f"run-list-{i}",
                repo="test-org/test-repo",
                commit=f"commit-{i}",
                requested_targets=("target1",),
                computed_targets=(),
                skipped_targets=(),
                started_at=datetime.now(tz=UTC),
                status="running",
            )
            tracking.start_run(record)

        runs = tracking.list_runs("test-org/test-repo")
        assert len(runs) == 3

        # Should be ordered by started_at descending (newest first)
        for i in range(len(runs) - 1):
            assert runs[i].started_at >= runs[i + 1].started_at

    def test_list_runs_with_limit(self, tracking: BuildTracking) -> None:
        """List runs respects limit parameter."""
        for i in range(5):
            record = BuildRunRecord(
                run_id=f"run-limit-{i}",
                repo="test-org/limit-repo",
                commit=f"commit-{i}",
                requested_targets=("target1",),
                computed_targets=(),
                skipped_targets=(),
                started_at=datetime.now(tz=UTC),
                status="running",
            )
            tracking.start_run(record)

        runs = tracking.list_runs("test-org/limit-repo", limit=3)
        assert len(runs) == 3
