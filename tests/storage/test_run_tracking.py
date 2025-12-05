"""Tests for run_tracking module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.runtime import RunContext
from tests._helpers.factories import make_snapshot
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.tracking import (
    PipelineRunRecord,
    PipelineRunTracking,
    PipelineStepRecord,
    StepCompletionParams,
)


def test_pipeline_run_record_stores_fields() -> None:
    """Verify PipelineRunRecord stores all fields."""
    now = datetime.now(tz=UTC)
    record = PipelineRunRecord(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        kind="full",
        trigger="cli",
        status="running",
        started_at=now,
    )

    assert record.run_id == "run-123"
    assert record.repo == "test/repo"
    assert record.status == "running"


def test_pipeline_step_record_stores_fields() -> None:
    """Verify PipelineStepRecord stores all fields."""
    now = datetime.now(tz=UTC)
    record = PipelineStepRecord(
        run_id="run-123",
        module="ingestion",
        stage="scan",
        name="file_scanner",
        status="succeeded",
        started_at=now,
    )

    assert record.run_id == "run-123"
    assert record.module == "ingestion"
    assert record.status == "succeeded"


def test_step_completion_params_to_record() -> None:
    """Verify StepCompletionParams.to_record creates PipelineStepRecord."""
    now = datetime.now(tz=UTC)
    params = StepCompletionParams(
        run_id="run-123",
        module="analytics",
        stage="compute",
        name="metrics_plugin",
        status="succeeded",
        started_at=now,
        row_counts={"analytics.metrics": 100},
    )

    record = params.to_record()

    assert isinstance(record, PipelineStepRecord)
    assert record.run_id == "run-123"
    assert record.status == "succeeded"
    assert record.completed_at is not None


def _make_run_context(run_id: str, tmp_path: Path) -> RunContext:
    """
    Create a RunContext for testing.

    Parameters
    ----------
    run_id
        Unique run identifier.
    tmp_path
        Temporary path for repo root.

    Returns
    -------
    RunContext
        A RunContext configured for testing.
    """
    snapshot = make_snapshot(repo="test/repo", commit="abc123", repo_root=tmp_path)
    return RunContext(
        run_id=run_id,
        kind="full",
        snapshot=snapshot,
        trigger="cli",
    )


def test_start_run_inserts_record(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify start_run inserts a pipeline run record."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    ctx = _make_run_context(run_id="run-test-1", tmp_path=tmp_path)

    tracking.start_run(ctx, pipeline_name="Test Pipeline")

    result = con.execute(
        "SELECT * FROM metadata.pipeline_runs WHERE run_id = ?",
        ["run-test-1"],
    ).fetchone()

    assert result is not None


def test_complete_run_updates_status(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify complete_run updates run status."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    ctx = _make_run_context(run_id="run-test-2", tmp_path=tmp_path)

    tracking.start_run(ctx)
    tracking.complete_run("run-test-2", status="succeeded")

    result = con.execute(
        "SELECT status FROM metadata.pipeline_runs WHERE run_id = ?",
        ["run-test-2"],
    ).fetchone()

    assert result is not None
    assert result[0] == "succeeded"


def test_complete_run_with_error_summary(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify complete_run stores error summary."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    ctx = _make_run_context(run_id="run-test-3", tmp_path=tmp_path)

    tracking.start_run(ctx)
    tracking.complete_run("run-test-3", status="failed", error_summary="Test error occurred")

    result = con.execute(
        "SELECT error_summary FROM metadata.pipeline_runs WHERE run_id = ?",
        ["run-test-3"],
    ).fetchone()

    assert result is not None
    assert result[0] == "Test error occurred"


def test_fetch_run_returns_record(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify fetch_run returns PipelineRunRecord."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    ctx = _make_run_context(run_id="run-test-4", tmp_path=tmp_path)

    tracking.start_run(ctx, pipeline_name="Fetch Test")

    record = tracking.fetch_run("run-test-4")

    assert record is not None
    assert isinstance(record, PipelineRunRecord)
    assert record.run_id == "run-test-4"


def test_fetch_run_returns_none_for_missing(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_run returns None for missing run."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)

    record = tracking.fetch_run("nonexistent-run")

    assert record is None


def test_record_step_inserts_step(fresh_gateway: StorageGateway) -> None:
    """Verify record_step inserts a step record."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    now = datetime.now(tz=UTC)

    step = PipelineStepRecord(
        run_id="run-step-1",
        module="ingestion",
        stage="scan",
        name="file_scanner",
        status="succeeded",
        started_at=now,
        completed_at=now,
        row_counts={"core.modules": 10},
        extra={"files_scanned": 5},
    )

    tracking.record_step(step)

    result = con.execute(
        """
        SELECT run_id, module, name, status
        FROM metadata.pipeline_steps
        WHERE run_id = ? AND name = ?
        """,
        ["run-step-1", "file_scanner"],
    ).fetchone()

    assert result is not None


def test_fetch_steps_returns_list(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_steps returns list of PipelineStepRecord."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    now = datetime.now(tz=UTC)

    step1 = PipelineStepRecord(
        run_id="run-fetch-steps",
        module="ingestion",
        stage="scan",
        name="scanner1",
        status="succeeded",
        started_at=now,
    )
    step2 = PipelineStepRecord(
        run_id="run-fetch-steps",
        module="analytics",
        stage="compute",
        name="metrics",
        status="succeeded",
        started_at=now,
    )

    tracking.record_step(step1)
    tracking.record_step(step2)

    steps = tracking.fetch_steps("run-fetch-steps")

    assert isinstance(steps, list)
    expected_step_count = 2
    assert len(steps) == expected_step_count


def test_fetch_steps_orders_by_module_stage_name(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_steps orders results by module, stage, name."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    now = datetime.now(tz=UTC)

    step_a = PipelineStepRecord(
        run_id="run-order-test",
        module="ingestion",
        stage="scan",
        name="z_scanner",
        status="succeeded",
        started_at=now,
    )
    step_b = PipelineStepRecord(
        run_id="run-order-test",
        module="analytics",
        stage="compute",
        name="a_metrics",
        status="succeeded",
        started_at=now,
    )

    tracking.record_step(step_a)
    tracking.record_step(step_b)

    steps = tracking.fetch_steps("run-order-test")

    assert steps[0].module == "analytics"


def test_start_step_returns_timestamp(fresh_gateway: StorageGateway) -> None:
    """Verify start_step returns started_at timestamp."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)

    started_at = tracking.start_step(
        run_id="run-start-step",
        module="graphs",
        stage="build",
        name="graph_builder",
    )

    assert isinstance(started_at, datetime)
    assert started_at.tzinfo is not None


def test_complete_step_updates_step(fresh_gateway: StorageGateway) -> None:
    """Verify complete_step updates step record."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)

    started_at = tracking.start_step(
        run_id="run-complete-step",
        module="analytics",
        stage="compute",
        name="test_plugin",
    )

    params = StepCompletionParams(
        run_id="run-complete-step",
        module="analytics",
        stage="compute",
        name="test_plugin",
        status="succeeded",
        started_at=started_at,
        row_counts={"analytics.test": 50},
    )

    tracking.complete_step(params)

    result = con.execute(
        """
        SELECT status, row_counts
        FROM metadata.pipeline_steps
        WHERE run_id = ? AND name = ?
        """,
        ["run-complete-step", "test_plugin"],
    ).fetchone()

    assert result is not None
    assert result[0] == "succeeded"


def test_record_step_with_none_row_counts(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify record_step handles None row_counts."""
    con = fresh_gateway.con
    tracking = PipelineRunTracking(con=con)
    now = datetime.now(tz=UTC)

    step = PipelineStepRecord(
        run_id="run-none-counts",
        module="ingestion",
        stage="scan",
        name="scanner",
        status="succeeded",
        started_at=now,
        row_counts=None,
        extra=None,
    )

    tracking.record_step(step)

    result = con.execute(
        "SELECT row_counts FROM metadata.pipeline_steps WHERE run_id = ?",
        ["run-none-counts"],
    ).fetchone()

    assert result is not None
