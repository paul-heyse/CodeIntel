"""Tests for ingest run tracking and sinks.

This module tests IngestRun, classify_error, and various sink implementations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from codeintel.ingestion.core.runs import (
    DuckDBIngestRunSink,
    IngestRun,
    IngestRunMode,
    IngestRunSink,
    IngestRunStatus,
    JsonlIngestRunSink,
    MultiSink,
    classify_error,
)
from codeintel.ingestion.tools.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
)
from codeintel.storage.gateway import DuckDBError, StorageGateway

# Test constants for magic values
TEST_TOTAL_MODULES = 50
TEST_TOTAL_FILES = 100
TEST_PROCESSED_FILES = 10
TEST_FAILED_FILES = 2
TEST_ELAPSED_SECONDS = 0.1
TEST_AVERAGE_TIME = 0.02
EXPECTED_LINE_COUNT = 2


# --- IngestRunStatus Tests ---


def test_ingest_run_status_values() -> None:
    """IngestRunStatus should have expected values."""
    assert IngestRunStatus.OK == "ok"
    assert IngestRunStatus.SKIPPED == "skipped"
    assert IngestRunStatus.ERROR == "error"


# --- IngestRunMode Tests ---


def test_ingest_run_mode_values() -> None:
    """IngestRunMode should have expected values."""
    assert IngestRunMode.FULL == "full"
    assert IngestRunMode.INCREMENTAL == "incremental"
    assert IngestRunMode.UNKNOWN == "unknown"


# --- IngestRun Tests ---


def test_ingest_run_create_minimal() -> None:
    """IngestRun should be creatable with minimal required fields."""
    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    assert run.run_id == "run-123"
    assert run.repo == "test/repo"
    assert run.commit == "abc123"
    assert run.step == "test_step"
    assert run.datasets == ("core.modules",)
    assert run.mode == IngestRunMode.FULL
    assert run.started_at == started
    assert run.finished_at is None
    assert run.duration_s is None
    assert run.status == IngestRunStatus.OK


def test_ingest_run_create_with_metrics() -> None:
    """IngestRun should accept row metrics."""
    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
        rows_before={"core.modules": 100},
        rows_after={"core.modules": 150},
        rows_inserted=TEST_TOTAL_MODULES,
        rows_deleted=0,
    )

    assert run.rows_before == {"core.modules": 100}
    assert run.rows_after == {"core.modules": 150}
    assert run.rows_inserted == TEST_TOTAL_MODULES
    assert run.rows_deleted == 0


def test_ingest_run_create_with_error() -> None:
    """IngestRun should accept error information."""
    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
        status=IngestRunStatus.ERROR,
        error_kind="parse_error",
        error_message="Failed to parse module",
    )

    assert run.status == IngestRunStatus.ERROR
    assert run.error_kind == "parse_error"
    assert run.error_message == "Failed to parse module"


def test_ingest_run_create_with_incremental_metrics() -> None:
    """IngestRun should accept incremental view metrics."""
    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.INCREMENTAL,
        started_at=started,
        modules_total=TEST_TOTAL_FILES,
        modules_changed=TEST_PROCESSED_FILES,
        modules_deleted=TEST_FAILED_FILES,
        modules_changed_ratio=TEST_ELAPSED_SECONDS,
        modules_deleted_ratio=TEST_AVERAGE_TIME,
        use_full_rebuild=False,
    )

    assert run.modules_total == TEST_TOTAL_FILES
    assert run.modules_changed == TEST_PROCESSED_FILES
    assert run.modules_deleted == TEST_FAILED_FILES
    assert run.modules_changed_ratio == TEST_ELAPSED_SECONDS
    assert run.modules_deleted_ratio == TEST_AVERAGE_TIME
    assert run.use_full_rebuild is False


def test_ingest_run_to_row_tuple() -> None:
    """IngestRun.to_row_tuple should produce valid tuple."""
    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    row = run.to_row_tuple()

    assert isinstance(row, tuple)
    assert len(row) > 0


# --- ClassifyError Tests ---


def test_classify_error_tool_not_found_error() -> None:
    """classify_error should classify ToolNotFoundError."""
    exc = ToolNotFoundError(ToolName.PYRIGHT, "/usr/bin/pyright")

    result = classify_error(exc)

    assert result == "tool_not_found"


def test_classify_error_tool_execution_error() -> None:
    """classify_error should classify ToolExecutionError."""
    mock_result = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("check", "."),
        returncode=1,
        stdout="",
        stderr="Error",
        duration_s=1.0,
    )
    exc = ToolExecutionError(mock_result)

    result = classify_error(exc)

    assert result == "tool_execution"


def test_classify_error_tool_execution_timeout() -> None:
    """classify_error should classify timeout errors specially."""
    mock_result = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("check", "."),
        returncode=1,
        stdout="",
        stderr="Process timed out",
        duration_s=30.0,
    )
    exc = ToolExecutionError(mock_result)

    result = classify_error(exc)

    assert result == "tool_timeout"


def test_classify_error_duckdb_error() -> None:
    """classify_error should classify DuckDBError."""
    exc = DuckDBError("Query failed")

    result = classify_error(exc)

    assert result == "db_error"


def test_classify_error_value_error() -> None:
    """classify_error should classify ValueError as parse_error."""
    exc = ValueError("Invalid format")

    result = classify_error(exc)

    assert result == "parse_error"


def test_classify_error_other_error() -> None:
    """classify_error should return exception class name for unknown errors."""
    exc = TypeError("Type mismatch")

    result = classify_error(exc)

    assert result == "TypeError"


# --- JsonlIngestRunSink Tests ---


def test_jsonl_sink_record_creates_file(tmp_path: Path) -> None:
    """JsonlIngestRunSink.record should create JSONL file."""
    log_path = tmp_path / "logs" / "ingest_runs.jsonl"
    sink = JsonlIngestRunSink(path=log_path)

    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    sink.record(run)

    assert log_path.exists()


def test_jsonl_sink_record_appends_jsonl(tmp_path: Path) -> None:
    """JsonlIngestRunSink.record should append JSONL lines."""
    log_path = tmp_path / "ingest_runs.jsonl"
    sink = JsonlIngestRunSink(path=log_path)

    started = datetime.now(tz=UTC)
    run1 = IngestRun(
        run_id="run-1",
        repo="test/repo",
        commit="abc123",
        step="step1",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )
    run2 = IngestRun(
        run_id="run-2",
        repo="test/repo",
        commit="abc123",
        step="step2",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    sink.record(run1)
    sink.record(run2)

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == EXPECTED_LINE_COUNT

    # Verify JSON is valid
    record1 = json.loads(lines[0])
    record2 = json.loads(lines[1])
    assert record1["run_id"] == "run-1"
    assert record2["run_id"] == "run-2"


def test_jsonl_sink_record_includes_timestamps(tmp_path: Path) -> None:
    """JsonlIngestRunSink should format timestamps as ISO."""
    log_path = tmp_path / "ingest_runs.jsonl"
    sink = JsonlIngestRunSink(path=log_path)

    started = datetime.now(tz=UTC)
    finished = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
        finished_at=finished,
    )

    sink.record(run)

    lines = log_path.read_text().strip().split("\n")
    record = json.loads(lines[0])
    assert "started_at" in record
    assert "finished_at" in record


# --- DuckDBIngestRunSink Tests ---


def test_duckdb_sink_record_inserts_row(fresh_gateway: StorageGateway) -> None:
    """DuckDBIngestRunSink.record should insert into core.ingest_runs."""
    sink = DuckDBIngestRunSink(gateway=fresh_gateway)

    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    sink.record(run)

    # Verify the row was inserted using the connection directly
    result = fresh_gateway.con.execute(
        "SELECT run_id FROM core.ingest_runs WHERE run_id = 'run-123'"
    )
    rows = result.fetchall()
    assert len(rows) == 1


# --- MultiSink Tests ---


class FailingSink(IngestRunSink):
    """A sink that always fails for testing purposes.

    Raises
    ------
    RuntimeError
        Always raised when record is called, to test error handling in MultiSink.
    """

    @staticmethod
    def record(run: IngestRun) -> None:
        """Raise error on record.

        Parameters
        ----------
        run
            The ingest run to record.

        Raises
        ------
        RuntimeError
            Always raised to simulate sink failure.
        """
        _run_id = run.run_id  # Use the run to avoid unused parameter warning
        msg = "Sink failure"
        raise RuntimeError(msg)


def test_multi_sink_record_fans_out(tmp_path: Path) -> None:
    """MultiSink.record should send to all sinks."""
    path1 = tmp_path / "sink1.jsonl"
    path2 = tmp_path / "sink2.jsonl"

    sink1 = JsonlIngestRunSink(path=path1)
    sink2 = JsonlIngestRunSink(path=path2)
    multi_sink = MultiSink(sinks=[sink1, sink2])

    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    multi_sink.record(run)

    assert path1.exists()
    assert path2.exists()


def test_multi_sink_record_continues_on_sink_failure(tmp_path: Path) -> None:
    """MultiSink.record should continue even if one sink fails."""
    path = tmp_path / "success.jsonl"
    failing_sink = FailingSink()
    success_sink = JsonlIngestRunSink(path=path)

    # Failing sink first, then success sink
    multi_sink = MultiSink(sinks=[failing_sink, success_sink])

    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    # Should not raise, and success_sink should still get the record
    multi_sink.record(run)

    assert path.exists()


def test_multi_sink_empty_sinks() -> None:
    """MultiSink with no sinks should be no-op."""
    multi_sink = MultiSink(sinks=[])

    started = datetime.now(tz=UTC)
    run = IngestRun(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        step="test_step",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=started,
    )

    # Should not raise
    multi_sink.record(run)
