"""Tests for ingestion runtime executor types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from codeintel.core.plugins.types.result import BasePluginExecutionRecord
from codeintel.ingestion.plugins.protocol import IngestPluginResult
from codeintel.ingestion.runtime.executor import IngestPluginExecutionRecord


class TestIngestPluginExecutionRecord:
    """Test IngestPluginExecutionRecord functionality."""

    def test_inherits_from_base(self) -> None:
        """Test that IngestPluginExecutionRecord extends BasePluginExecutionRecord."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
        )

        assert isinstance(record, BasePluginExecutionRecord)

    def test_duration_s_inherited(self) -> None:
        """Test duration_s property is inherited from base."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = datetime(2024, 1, 1, 12, 0, 3, tzinfo=UTC)

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        # Inherited property from BasePluginExecutionRecord
        assert record.duration_s == 3.0

    def test_computed_duration_ms_inherited(self) -> None:
        """Test computed_duration_ms property is inherited from base."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = started + timedelta(milliseconds=1500)

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        # Inherited property from BasePluginExecutionRecord
        assert record.computed_duration_ms == 1500.0

    def test_result_field(self) -> None:
        """Test result field with IngestPluginResult."""
        result = IngestPluginResult.ok(row_counts={"table_a": 50})

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            result=result,
        )

        assert record.result is not None
        assert record.result.row_counts is not None
        assert record.result.row_counts["table_a"] == 50

    def test_error_field_exception(self) -> None:
        """Test error field stores Exception instances."""
        exc = ValueError("Invalid input")

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            error=exc,
        )

        assert record.error is exc
        assert isinstance(record.error, Exception)

    def test_rows_written_field(self) -> None:
        """Test rows_written field."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            rows_written=100,
        )

        assert record.rows_written == 100

    def test_table_counts_field(self) -> None:
        """Test table_counts field."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            table_counts={"table_a": 30, "table_b": 70},
        )

        assert record.table_counts["table_a"] == 30
        assert record.table_counts["table_b"] == 70

    def test_success_property_with_result(self) -> None:
        """Test success property returns True with result and no error."""
        result = IngestPluginResult.ok()

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            result=result,
        )

        assert record.success is True

    def test_success_property_with_error(self) -> None:
        """Test success property returns False with error."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            error=ValueError("oops"),
        )

        assert record.success is False

    def test_success_property_no_result(self) -> None:
        """Test success property returns False with no result."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
        )

        assert record.success is False

    def test_status_property_succeeded(self) -> None:
        """Test status property returns 'succeeded' with result."""
        result = IngestPluginResult.ok()

        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            result=result,
        )

        assert record.status == "succeeded"

    def test_status_property_failed(self) -> None:
        """Test status property returns 'failed' with error."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            error=ValueError("error"),
        )

        assert record.status == "failed"

    def test_status_property_skipped(self) -> None:
        """Test status property returns 'skipped' with no result or error."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
        )

        assert record.status == "skipped"

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        record = IngestPluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
        )

        assert record.result is None
        assert record.error is None
        assert record.rows_written == 0
        assert record.table_counts == {}
        assert record.ended_at is None
