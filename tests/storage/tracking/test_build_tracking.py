"""Unit tests for BuildTracking storage accessor."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.manifest import BuildRunRecord, OutputManifest
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.tracking.build_tracking import BuildTracking

MANIFEST_COUNT = 3
UPDATED_DURATION_MS = 2000.0
LIST_RUN_COUNT = 3
RUN_LIMIT = 3


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

    @staticmethod
    def test_save_and_load_manifest(tracking: BuildTracking) -> None:
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

        loaded_manifest = expect_is_not_none(loaded)
        expect_equal(loaded_manifest.target, manifest.target)
        expect_equal(loaded_manifest.repo, manifest.repo)
        expect_equal(loaded_manifest.commit, manifest.commit)
        expect_equal(loaded_manifest.plugin, manifest.plugin)
        expect_equal(loaded_manifest.duration_ms, manifest.duration_ms)
        expect_equal(loaded_manifest.input_hash, manifest.input_hash)
        expect_equal(loaded_manifest.output_hash, manifest.output_hash)
        expect_equal(loaded_manifest.row_count, manifest.row_count)
        expect_equal(loaded_manifest.options_hash, manifest.options_hash)

    @staticmethod
    def test_load_nonexistent_manifest(tracking: BuildTracking) -> None:
        """Loading non-existent manifest returns None."""
        loaded = tracking.load_manifest("nonexistent", "no/repo", "nocommit")
        expect_is_none(loaded)

    @staticmethod
    def test_upsert_manifest(tracking: BuildTracking) -> None:
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
        loaded_manifest = expect_is_not_none(loaded)
        expect_equal(loaded_manifest.duration_ms, UPDATED_DURATION_MS)
        expect_equal(loaded_manifest.input_hash, "input2")

    @staticmethod
    def test_list_manifests(tracking: BuildTracking) -> None:
        """List all manifests for a repo/commit."""
        now = datetime.now(tz=UTC)
        for i in range(MANIFEST_COUNT):
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
        expect_length(manifests, MANIFEST_COUNT)

        # Should be ordered by target name
        names = [m.target for m in manifests]
        expect_equal(names, sorted(names))

    @staticmethod
    def test_delete_manifests(tracking: BuildTracking) -> None:
        """Delete all manifests for a repo/commit."""
        now = datetime.now(tz=UTC)
        for i in range(MANIFEST_COUNT):
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

        # Verify manifests were created
        manifests_before = tracking.list_manifests("test-org/test-repo", "abc123")
        expect_true(len(manifests_before) > 0)

        # Delete all manifests
        tracking.delete_manifests("test-org/test-repo", "abc123")

        # Verify all manifests were deleted
        manifests_after = tracking.list_manifests("test-org/test-repo", "abc123")
        expect_length(manifests_after, 0)


class TestBuildRunOperations:
    """Tests for BuildRunRecord CRUD operations."""

    @staticmethod
    def test_start_and_fetch_run(tracking: BuildTracking) -> None:
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

        loaded_run = expect_is_not_none(loaded)
        expect_equal(loaded_run.run_id, "run-123")
        expect_equal(loaded_run.repo, "test-org/test-repo")
        expect_equal(loaded_run.commit, "abc123")
        expect_equal(loaded_run.requested_targets, ("target1", "target2"))
        expect_equal(loaded_run.status, "running")

    @staticmethod
    def test_fetch_nonexistent_run(tracking: BuildTracking) -> None:
        """Fetching non-existent run returns None."""
        loaded = tracking.fetch_run("nonexistent-run")
        expect_is_none(loaded)

    @staticmethod
    def test_complete_run_succeeded(tracking: BuildTracking) -> None:
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
        loaded_run = expect_is_not_none(loaded)
        expect_equal(loaded_run.status, "succeeded")
        expect_equal(loaded_run.computed_targets, ("target1", "target2"))
        expect_equal(loaded_run.skipped_targets, ("target3",))
        expect_is_not_none(loaded_run.completed_at)
        expect_is_not_none(loaded_run.duration_ms)

    @staticmethod
    def test_complete_run_failed(tracking: BuildTracking) -> None:
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
        loaded_run = expect_is_not_none(loaded)
        expect_equal(loaded_run.status, "failed")
        expect_equal(loaded_run.error_summary, "Test error occurred")

    @staticmethod
    def test_list_runs(tracking: BuildTracking) -> None:
        """List recent runs for a repository."""
        for i in range(LIST_RUN_COUNT):
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
        expect_length(runs, LIST_RUN_COUNT)

        # Should be ordered by started_at descending (newest first)
        for i in range(len(runs) - 1):
            expect_true(runs[i].started_at >= runs[i + 1].started_at)

    @staticmethod
    def test_list_runs_with_limit(tracking: BuildTracking) -> None:
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

        runs = tracking.list_runs("test-org/limit-repo", limit=RUN_LIMIT)
        expect_length(runs, RUN_LIMIT)
