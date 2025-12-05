"""Tests for plugin result types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from codeintel.core.plugins.types.result import (
    BasePluginExecutionRecord,
    BasePluginResult,
    PluginExecutionRecord,
    PluginResult,
)


class TestBasePluginExecutionRecord:
    """Test BasePluginExecutionRecord functionality."""

    def test_duration_s_with_ended_at(self) -> None:
        """Test duration_s property computes correctly when ended_at is set."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC)

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        assert record.duration_s == 5.0

    def test_duration_s_without_ended_at(self) -> None:
        """Test duration_s returns 0.0 when ended_at is None."""
        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            ended_at=None,
        )

        assert record.duration_s == 0.0

    def test_computed_duration_ms(self) -> None:
        """Test computed_duration_ms returns duration in milliseconds."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = datetime(2024, 1, 1, 12, 0, 2, 500000, tzinfo=UTC)  # 2.5 seconds

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        assert record.computed_duration_ms == 2500.0

    def test_computed_duration_ms_without_ended_at(self) -> None:
        """Test computed_duration_ms returns 0.0 when ended_at is None."""
        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            ended_at=None,
        )

        assert record.computed_duration_ms == 0.0

    def test_fractional_duration(self) -> None:
        """Test duration with fractional seconds."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = started + timedelta(milliseconds=1234)

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        assert record.duration_s == pytest.approx(1.234, rel=1e-3)
        assert record.computed_duration_ms == pytest.approx(1234.0, rel=1e-3)


class TestPluginExecutionRecordWithResult:
    """Test PluginExecutionRecord with result field populated."""

    def test_result_field_access(self) -> None:
        """Test that result field can be accessed for row_counts."""
        result = PluginResult.ok(
            row_counts={"table_a": 100, "table_b": 200},
        )
        started = datetime.now(tz=UTC)
        ended = started + timedelta(seconds=1)

        record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=started,
            ended_at=ended,
            duration_ms=1000.0,
            result=result,
        )

        assert record.result is not None
        assert record.result.row_counts is not None
        assert record.result.row_counts["table_a"] == 100
        assert record.result.row_counts["table_b"] == 200

    def test_result_field_none(self) -> None:
        """Test PluginExecutionRecord with no result."""
        started = datetime.now(tz=UTC)
        ended = started + timedelta(seconds=1)

        record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="failed",
            started_at=started,
            ended_at=ended,
            duration_ms=1000.0,
            result=None,
            error="Something went wrong",
        )

        assert record.result is None
        assert record.error == "Something went wrong"

    def test_row_counts_via_result(self) -> None:
        """Test accessing row_counts through result field (canonical pattern)."""
        result = PluginResult.ok(row_counts={"output.table": 42})
        started = datetime.now(tz=UTC)
        ended = started + timedelta(seconds=1)

        record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=started,
            ended_at=ended,
            duration_ms=1000.0,
            result=result,
        )

        # Canonical pattern for accessing row_counts
        row_counts = (
            dict(record.result.row_counts)
            if record.result and record.result.row_counts
            else None
        )

        assert row_counts is not None
        assert row_counts["output.table"] == 42


class TestBasePluginResult:
    """Test BasePluginResult factory methods."""

    def test_ok_factory(self) -> None:
        """Test ok() factory method."""
        result = BasePluginResult.ok(
            row_counts={"table": 10},
            input_hash="abc123",
        )

        assert result.success is True
        assert result.row_counts is not None
        assert result.row_counts["table"] == 10
        assert result.input_hash == "abc123"
        assert result.error is None
        assert result.skipped is False

    def test_fail_factory(self) -> None:
        """Test fail() factory method."""
        result = BasePluginResult.fail("Something broke", error_kind="validation")

        assert result.success is False
        assert result.error == "Something broke"
        assert result.error_kind == "validation"
        assert result.row_counts is None

    def test_skip_factory(self) -> None:
        """Test skip() factory method."""
        result = BasePluginResult.skip("Inputs unchanged")

        assert result.success is True
        assert result.skipped is True
        assert result.skip_reason == "Inputs unchanged"

    def test_status_property_succeeded(self) -> None:
        """Test status property for successful result."""
        result = BasePluginResult.ok()
        assert result.status == "succeeded"

    def test_status_property_failed(self) -> None:
        """Test status property for failed result."""
        result = BasePluginResult.fail("error")
        assert result.status == "failed"

    def test_status_property_skipped(self) -> None:
        """Test status property for skipped result."""
        result = BasePluginResult.skip("reason")
        assert result.status == "skipped"


class TestPluginResult:
    """Test PluginResult factory methods."""

    def test_ok_with_artifacts(self) -> None:
        """Test ok() with artifacts."""
        result = PluginResult.ok(
            row_counts={"table": 5},
            artifacts={"output_file": "/path/to/file"},
        )

        assert result.success is True
        assert result.artifacts["output_file"] == "/path/to/file"

    def test_fail_with_warnings(self) -> None:
        """Test fail() preserves warnings."""
        result = PluginResult.fail(
            "Critical error",
            warnings=("Warning 1", "Warning 2"),
        )

        assert result.success is False
        assert result.error == "Critical error"
        assert result.warnings == ("Warning 1", "Warning 2")

