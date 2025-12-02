"""Unit tests for run_registry module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.runtime import RunContext
from codeintel.storage.metadata_bootstrap import (
    PIPELINE_INDEXES_DDL,
    PIPELINE_RUNS_DDL,
    PIPELINE_STEPS_DDL,
)
from codeintel.storage.run_tracking import (
    PipelineRunRecord,
    PipelineRunTracking,
    PipelineStepRecord,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


@pytest.fixture
def test_con() -> PipelineRunTracking:
    """Create an in-memory DuckDB connection with pipeline tables.

    Returns
    -------
    PipelineRunTracking
        Pipeline run tracking accessor with in-memory database.
    """
    import duckdb  # noqa: PLC0415

    # Use in-memory database with just the pipeline tables
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA IF NOT EXISTS metadata")
    con.execute(PIPELINE_RUNS_DDL)
    con.execute(PIPELINE_STEPS_DDL)
    for index_stmt in PIPELINE_INDEXES_DDL.strip().split(";"):
        stripped_stmt = index_stmt.strip()
        if stripped_stmt:
            con.execute(stripped_stmt)
    return PipelineRunTracking(con)


@pytest.fixture
def sample_snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a sample snapshot for testing.

    Returns
    -------
    SnapshotRef
        Sample snapshot reference for tests.
    """
    return SnapshotRef(
        repo="github.com/demo/repo",
        commit="deadbeef" * 5,
        repo_root=tmp_path,
    )


@pytest.fixture
def sample_run_context(sample_snapshot: SnapshotRef) -> RunContext:
    """Create a sample RunContext for testing.

    Returns
    -------
    RunContext
        Sample run context for tests.
    """
    return RunContext(
        run_id="ci-123",
        kind="analytics",
        snapshot=sample_snapshot,
        trigger="cli",
        requested_operation="functions.summary",
        requested_datasets=("analytics.function_metrics",),
    )


class TestStartAndFetchRun:
    """Test start_run and fetch_run methods."""

    @staticmethod
    def test_start_run_creates_record(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Starting a run should create a record in the database."""
        test_con.start_run(
            sample_run_context,
            pipeline_name="analytics:full",
        )

        rec = test_con.fetch_run("ci-123")
        assert rec is not None
        assert rec.run_id == "ci-123"
        assert rec.repo == "github.com/demo/repo"
        assert rec.commit == "deadbeef" * 5
        assert rec.kind == "analytics"
        assert rec.trigger == "cli"
        assert rec.status == "running"
        assert rec.pipeline_name == "analytics:full"
        assert rec.requested_operation == "functions.summary"
        assert rec.requested_datasets == ("analytics.function_metrics",)

    @staticmethod
    def test_fetch_nonexistent_run(test_con: PipelineRunTracking) -> None:
        """Fetching a nonexistent run should return None."""
        rec = test_con.fetch_run("nonexistent")
        assert rec is None

    @staticmethod
    def test_start_run_replaces_existing(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Starting a run with the same ID should replace the existing record."""
        test_con.start_run(
            sample_run_context,
            pipeline_name="first",
        )

        test_con.start_run(
            sample_run_context,
            pipeline_name="second",
        )

        rec = test_con.fetch_run("ci-123")
        assert rec is not None
        assert rec.pipeline_name == "second"


class TestCompleteRun:
    """Test complete_run method."""

    @staticmethod
    def test_complete_run_updates_status(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Completing a run should update status and completion time."""
        test_con.start_run(sample_run_context)

        test_con.complete_run(
            "ci-123",
            status="succeeded",
        )

        rec = test_con.fetch_run("ci-123")
        assert rec is not None
        assert rec.status == "succeeded"
        assert rec.completed_at is not None
        assert rec.error_summary is None

    @staticmethod
    def test_complete_run_with_error(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Completing a run with error should record error summary."""
        test_con.start_run(sample_run_context)

        test_con.complete_run(
            "ci-123",
            status="failed",
            error_summary="Plugin X failed with error Y",
        )

        rec = test_con.fetch_run("ci-123")
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_summary == "Plugin X failed with error Y"


class TestRecordStep:
    """Test record_step and fetch_steps methods."""

    @staticmethod
    def test_record_step_creates_record(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Recording a step should create a record in the database."""
        test_con.start_run(sample_run_context)

        now = datetime.now(tz=UTC)
        test_con.record_step(
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

        steps = test_con.fetch_steps("ci-123")
        assert len(steps) == 1
        step = steps[0]
        assert step.module == "ingestion"
        assert step.stage == "scan"
        assert step.name == "repo_scan"
        assert step.status == "succeeded"
        assert step.row_counts == {"core.modules": 10}
        assert step.extra == {"note": "ok"}

    @staticmethod
    def test_fetch_steps_empty(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Fetching steps for a run with no steps should return empty list."""
        test_con.start_run(sample_run_context)
        steps = test_con.fetch_steps("ci-123")
        assert steps == []

    @staticmethod
    def test_record_step_replaces_existing(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Recording a step with same key should replace the existing record."""
        test_con.start_run(sample_run_context)

        now = datetime.now(tz=UTC)
        test_con.record_step(
            PipelineStepRecord(
                run_id="ci-123",
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="running",
                started_at=now,
            ),
        )

        test_con.record_step(
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

        steps = test_con.fetch_steps("ci-123")
        assert len(steps) == 1
        assert steps[0].status == "succeeded"
        assert steps[0].row_counts == {"core.modules": 10}


class TestStartAndCompleteStep:
    """Test start_step and complete_step convenience methods."""

    @staticmethod
    def test_start_step_creates_running_record(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Start step should create a record with running status."""
        test_con.start_run(sample_run_context)

        started_at = test_con.start_step(
            run_id="ci-123",
            module="graphs",
            stage="build",
            name="call_graph_builder",
        )

        assert started_at is not None
        steps = test_con.fetch_steps("ci-123")
        assert len(steps) == 1
        assert steps[0].status == "running"
        assert steps[0].completed_at is None

    @staticmethod
    def test_complete_step_updates_record(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """Complete step should update the record with final status."""
        test_con.start_run(sample_run_context)

        started_at = test_con.start_step(
            run_id="ci-123",
            module="analytics",
            stage="function",
            name="function_metrics",
        )

        test_con.complete_step(
            run_id="ci-123",
            module="analytics",
            stage="function",
            name="function_metrics",
            status="succeeded",
            started_at=started_at,
            row_counts={"analytics.function_metrics": 100},
        )

        steps = test_con.fetch_steps("ci-123")
        assert len(steps) == 1
        assert steps[0].status == "succeeded"
        assert steps[0].completed_at is not None
        assert steps[0].row_counts == {"analytics.function_metrics": 100}


class TestMultipleSteps:
    """Test runs with multiple steps."""

    @staticmethod
    def test_run_with_multiple_steps(
        test_con: PipelineRunTracking,
        sample_run_context: RunContext,
    ) -> None:
        """A run should support multiple steps from different modules."""
        test_con.start_run(sample_run_context)

        now = datetime.now(tz=UTC)

        # Add ingestion step
        test_con.record_step(
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
        test_con.record_step(
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
        test_con.record_step(
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

        steps = test_con.fetch_steps("ci-123")
        expected_step_count = 3
        assert len(steps) == expected_step_count

        # Steps are ordered by module, stage, name
        modules = [s.module for s in steps]
        assert modules == ["analytics", "graphs", "ingestion"]

        # Check status distribution
        statuses = {s.module: s.status for s in steps}
        assert statuses["ingestion"] == "succeeded"
        assert statuses["graphs"] == "succeeded"
        assert statuses["analytics"] == "failed"


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

        with pytest.raises(AttributeError):
            setattr(record, "status", "succeeded")  # noqa: B010 - testing frozen dataclass

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

        with pytest.raises(AttributeError):
            setattr(record, "status", "succeeded")  # noqa: B010 - testing frozen dataclass
