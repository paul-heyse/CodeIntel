"""Tests for ingest run tracking and sinks.

This module tests IngestRun, classify_error, and various sink implementations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunResult,
)
from codeintel.ingestion.ingest_runs import (
    DuckDBIngestRunSink,
    IngestRun,
    IngestRunMode,
    IngestRunSink,
    IngestRunStatus,
    JsonlIngestRunSink,
    MultiSink,
    classify_error,
)
from codeintel.storage.gateway import DuckDBError, StorageGateway


class TestIngestRunStatus:
    """Tests for IngestRunStatus enum."""

    def test_status_values(self) -> None:
        """IngestRunStatus should have expected values."""
        assert IngestRunStatus.OK == "ok"
        assert IngestRunStatus.SKIPPED == "skipped"
        assert IngestRunStatus.ERROR == "error"


class TestIngestRunMode:
    """Tests for IngestRunMode enum."""

    def test_mode_values(self) -> None:
        """IngestRunMode should have expected values."""
        assert IngestRunMode.FULL == "full"
        assert IngestRunMode.INCREMENTAL == "incremental"
        assert IngestRunMode.UNKNOWN == "unknown"


class TestIngestRun:
    """Tests for IngestRun dataclass."""

    def test_create_minimal(self) -> None:
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

    def test_create_with_metrics(self) -> None:
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
            rows_inserted=50,
            rows_deleted=0,
        )

        assert run.rows_before == {"core.modules": 100}
        assert run.rows_after == {"core.modules": 150}
        assert run.rows_inserted == 50
        assert run.rows_deleted == 0

    def test_create_with_error(self) -> None:
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

    def test_create_with_incremental_metrics(self) -> None:
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
            modules_total=100,
            modules_changed=10,
            modules_deleted=2,
            modules_changed_ratio=0.1,
            modules_deleted_ratio=0.02,
            use_full_rebuild=False,
        )

        assert run.modules_total == 100
        assert run.modules_changed == 10
        assert run.modules_deleted == 2
        assert run.modules_changed_ratio == 0.1
        assert run.modules_deleted_ratio == 0.02
        assert run.use_full_rebuild is False

    def test_to_row_tuple(self) -> None:
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


class TestClassifyError:
    """Tests for classify_error function."""

    def test_tool_not_found_error(self) -> None:
        """classify_error should classify ToolNotFoundError."""
        from codeintel.ingestion.infrastructure_utilities.tool_runner import (  # noqa: PLC0415
            ToolName,
        )

        exc = ToolNotFoundError(ToolName.PYRIGHT, "/usr/bin/pyright")

        result = classify_error(exc)

        assert result == "tool_not_found"

    def test_tool_execution_error(self) -> None:
        """classify_error should classify ToolExecutionError."""
        from codeintel.ingestion.infrastructure_utilities.tool_runner import (  # noqa: PLC0415
            ToolName,
        )

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

    def test_tool_execution_timeout(self) -> None:
        """classify_error should classify timeout errors specially."""
        from codeintel.ingestion.infrastructure_utilities.tool_runner import (  # noqa: PLC0415
            ToolName,
        )

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

    def test_duckdb_error(self) -> None:
        """classify_error should classify DuckDBError."""
        exc = DuckDBError("Query failed")

        result = classify_error(exc)

        assert result == "db_error"

    def test_value_error(self) -> None:
        """classify_error should classify ValueError as parse_error."""
        exc = ValueError("Invalid format")

        result = classify_error(exc)

        assert result == "parse_error"

    def test_other_error(self) -> None:
        """classify_error should return exception class name for unknown errors."""
        exc = TypeError("Type mismatch")

        result = classify_error(exc)

        assert result == "TypeError"


class TestJsonlIngestRunSink:
    """Tests for JsonlIngestRunSink."""

    def test_record_creates_file(self, tmp_path: Path) -> None:
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

    def test_record_appends_jsonl(self, tmp_path: Path) -> None:
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
        assert len(lines) == 2

        # Verify JSON is valid
        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])
        assert record1["run_id"] == "run-1"
        assert record2["run_id"] == "run-2"

    def test_record_includes_timestamps(self, tmp_path: Path) -> None:
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


class TestDuckDBIngestRunSink:
    """Tests for DuckDBIngestRunSink."""

    def test_record_inserts_row(self, fresh_gateway: StorageGateway) -> None:
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


class TestMultiSink:
    """Tests for MultiSink."""

    def test_record_fans_out(self, tmp_path: Path) -> None:
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

    def test_record_continues_on_sink_failure(self, tmp_path: Path) -> None:
        """MultiSink.record should continue even if one sink fails."""

        class FailingSink(IngestRunSink):
            """Sink that always fails."""

            def record(self, run: IngestRun) -> None:
                """Raise error on record."""
                assert run.run_id  # Ensure the record is passed through
                msg = "Sink failure"
                raise RuntimeError(msg)

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

    def test_empty_sinks(self) -> None:
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
