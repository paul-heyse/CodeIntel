"""Unit tests for run_registry module."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.execution import RunContext
from codeintel.storage.tracking import (
    PipelineRunRecord,
    PipelineStepRecord,
    StepCompletionParams,
)
from tests._helpers.assertions import assert_cannot_setattr, expect_equal, expect_true
from tests._helpers.run_tracking import (
    ExpectedRun,
    RunContextOptions,
    RunTrackingHarness,
    expect_run,
    expect_steps,
)


@pytest.fixture
def sample_run_context(run_tracking_harness: RunTrackingHarness) -> RunContext:
    """
    Create a sample RunContext for testing.

    Returns
    -------
    RunContext
        Run context with preset identifiers.
    """
    options = RunContextOptions(
        repo="github.com/demo/repo",
        commit="deadbeef" * 5,
        kind="analytics",
        trigger="cli",
        requested_operation="functions.summary",
        requested_datasets=("analytics.function_metrics",),
    )
    return run_tracking_harness.make_context(run_id="ci-123", options=options)


class TestStartAndFetchRun:
    """Test start_run and fetch_run methods."""

    @staticmethod
    def test_start_run_creates_record(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Starting a run should create a record in the database."""
        run_tracking_harness.tracking.start_run(sample_run_context, pipeline_name="analytics:full")

        rec = expect_run(
            run_tracking_harness.tracking.fetch_run("ci-123"),
            ExpectedRun(
                run_id="ci-123",
                repo="github.com/demo/repo",
                status="running",
                kind="analytics",
                trigger="cli",
                pipeline_name="analytics:full",
                requested_operation="functions.summary",
                requested_datasets=("analytics.function_metrics",),
            ),
        )
        expect_equal(rec.commit, "deadbeef" * 5)

    @staticmethod
    def test_fetch_nonexistent_run(run_tracking_harness: RunTrackingHarness) -> None:
        """Fetching a nonexistent run should return None."""
        rec = run_tracking_harness.tracking.fetch_run("nonexistent")
        expect_true(rec is None, message="Expected no run record for nonexistent run.")

    @staticmethod
    def test_start_run_replaces_existing(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Starting a run with the same ID should replace the existing record."""
        run_tracking_harness.tracking.start_run(
            sample_run_context,
            pipeline_name="first",
        )

        run_tracking_harness.tracking.start_run(
            sample_run_context,
            pipeline_name="second",
        )

        rec = expect_run(
            run_tracking_harness.tracking.fetch_run("ci-123"),
            ExpectedRun(pipeline_name="second"),
        )
        expect_equal(rec.pipeline_name, "second")


class TestCompleteRun:
    """Test complete_run method."""

    @staticmethod
    def test_complete_run_updates_status(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Completing a run should update status and completion time."""
        run_tracking_harness.tracking.start_run(sample_run_context)

        run_tracking_harness.tracking.complete_run(
            "ci-123",
            status="succeeded",
        )

        rec = expect_run(
            run_tracking_harness.tracking.fetch_run("ci-123"),
            ExpectedRun(status="succeeded"),
        )
        expect_true(rec.completed_at is not None, message="Expected completed_at to be set.")
        expect_true(rec.error_summary is None, message="Expected no error summary on success.")

    @staticmethod
    def test_complete_run_with_error(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Completing a run with error should record error summary."""
        run_tracking_harness.tracking.start_run(sample_run_context)

        run_tracking_harness.tracking.complete_run(
            "ci-123",
            status="failed",
            error_summary="Plugin X failed with error Y",
        )

        expect_run(
            run_tracking_harness.tracking.fetch_run("ci-123"),
            ExpectedRun(status="failed", error_summary="Plugin X failed with error Y"),
        )


class TestRecordStep:
    """Test record_step and fetch_steps methods."""

    @staticmethod
    def test_record_step_creates_record(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Recording a step should create a record in the database."""
        run_tracking_harness.tracking.start_run(sample_run_context)

        now = datetime.now(tz=UTC)
        run_tracking_harness.tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"core.modules": 10},
                extra={"note": "ok"},
            ),
        )

        steps = expect_steps(run_tracking_harness.tracking.fetch_steps("ci-123"), expected_count=1)
        step = steps[0]
        expect_equal(step.module, "ingestion")
        expect_equal(step.stage, "scan")
        expect_equal(step.name, "repo_scan")
        expect_equal(step.status, "succeeded")
        expect_equal(step.row_counts, {"core.modules": 10})
        expect_equal(step.extra, {"note": "ok"})

    @staticmethod
    def test_fetch_steps_empty(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Fetching steps for a run with no steps should return empty list."""
        run_tracking_harness.tracking.start_run(sample_run_context)
        steps = run_tracking_harness.tracking.fetch_steps("ci-123")
        expect_equal(steps, [])

    @staticmethod
    def test_record_step_replaces_existing(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Recording a step with same key should replace the existing record."""
        tracking = run_tracking_harness.tracking
        tracking.start_run(sample_run_context)

        now = datetime.now(tz=UTC)
        tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="running",
                started_at=now,
            ),
        )

        tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"core.modules": 10},
            ),
        )

        steps = expect_steps(tracking.fetch_steps("ci-123"), expected_count=1)
        expect_equal(steps[0].status, "succeeded")
        expect_equal(steps[0].row_counts, {"core.modules": 10})


class TestStartAndCompleteStep:
    """Test start_step and complete_step convenience methods."""

    @staticmethod
    def test_start_step_creates_running_record(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Start step should create a record with running status."""
        tracking = run_tracking_harness.tracking
        tracking.start_run(sample_run_context)

        started_at = tracking.start_step(
            run_id="ci-123",
            module="graphs",
            stage="build",
            name="call_graph_builder",
        )

        expect_true(started_at is not None, message="Expected start time to be set.")
        steps = expect_steps(tracking.fetch_steps("ci-123"), expected_count=1)
        expect_equal(steps[0].status, "running")
        expect_true(steps[0].completed_at is None, message="Expected completed_at to be None.")

    @staticmethod
    def test_complete_step_updates_record(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """Complete step should update the record with final status."""
        tracking = run_tracking_harness.tracking
        tracking.start_run(sample_run_context)

        started_at = tracking.start_step(
            run_id="ci-123",
            module="analytics",
            stage="function",
            name="function_metrics",
        )

        tracking.complete_step(
            StepCompletionParams(
                run_id="ci-123",
                module="analytics",
                stage="function",
                name="function_metrics",
                status="succeeded",
                started_at=started_at,
                row_counts={"analytics.function_metrics": 100},
            )
        )

        steps = expect_steps(tracking.fetch_steps("ci-123"), expected_count=1)
        expect_equal(steps[0].status, "succeeded")
        expect_true(steps[0].completed_at is not None, message="Expected step completion time.")
        expect_equal(steps[0].row_counts, {"analytics.function_metrics": 100})


class TestMultipleSteps:
    """Test runs with multiple steps."""

    @staticmethod
    def test_run_with_multiple_steps(
        run_tracking_harness: RunTrackingHarness,
        sample_run_context: RunContext,
    ) -> None:
        """A run should support multiple steps from different modules."""
        tracking = run_tracking_harness.tracking
        tracking.start_run(sample_run_context)

        now = datetime.now(tz=UTC)

        # Add ingestion step
        tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="succeeded",
                started_at=now,
                completed_at=now,
            ),
        )

        # Add graphs step
        tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="graphs",
                stage="build",
                name="call_graph_builder",
                status="succeeded",
                started_at=now,
                completed_at=now,
            ),
        )

        # Add analytics step
        tracking.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="analytics",
                stage="function",
                name="function_metrics",
                status="failed",
                started_at=now,
                completed_at=now,
                extra={"error": "some error"},
            ),
        )

        steps = expect_steps(tracking.fetch_steps("ci-123"), expected_count=3)

        # Steps are ordered by module, stage, name
        modules = [s.module for s in steps]
        expect_equal(modules, ["analytics", "graphs", "ingestion"])

        # Check status distribution
        statuses = {s.module: s.status for s in steps}
        expect_equal(statuses["ingestion"], "succeeded")
        expect_equal(statuses["graphs"], "succeeded")
        expect_equal(statuses["analytics"], "failed")


class TestDataclasses:
    """Test dataclass properties."""

    @staticmethod
    def test_pipeline_run_record_frozen() -> None:
        """PipelineRunRecord should be frozen."""
        now = datetime.now(tz=UTC)
        record = PipelineRunRecord(
            run_id="test",
            repo="repo",
            commit="abc",
            kind="ingest",
            trigger="cli",
            status="running",
            started_at=now,
        )

        assert_cannot_setattr(record, "status", "succeeded")

    @staticmethod
    def test_pipeline_step_record_frozen() -> None:
        """PipelineStepRecord should be frozen."""
        now = datetime.now(tz=UTC)
        record = PipelineStepRecord(
            run_id="test",
            module="ingestion",
            stage="scan",
            name="plugin",
            status="running",
            started_at=now,
        )

        assert_cannot_setattr(record, "status", "succeeded")
