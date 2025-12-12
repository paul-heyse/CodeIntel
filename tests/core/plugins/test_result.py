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
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


class TestBasePluginExecutionRecord:
    """Test BasePluginExecutionRecord functionality."""

    @staticmethod
    def test_duration_s_with_ended_at() -> None:
        """Test duration_s property computes correctly when ended_at is set."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC)

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        expect_equal(record.duration_s, 5.0)

    @staticmethod
    def test_duration_s_without_ended_at() -> None:
        """Test duration_s returns 0.0 when ended_at is None."""
        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            ended_at=None,
        )

        expect_equal(record.duration_s, 0.0)

    @staticmethod
    def test_computed_duration_ms() -> None:
        """Test computed_duration_ms returns duration in milliseconds."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = datetime(2024, 1, 1, 12, 0, 2, 500000, tzinfo=UTC)

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        expect_equal(record.computed_duration_ms, 2500.0)

    @staticmethod
    def test_computed_duration_ms_without_ended_at() -> None:
        """Test computed_duration_ms returns 0.0 when ended_at is None."""
        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=datetime.now(tz=UTC),
            ended_at=None,
        )

        expect_equal(record.computed_duration_ms, 0.0)

    @staticmethod
    def test_fractional_duration() -> None:
        """Test duration with fractional seconds."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ended = started + timedelta(milliseconds=1234)

        record = BasePluginExecutionRecord(
            plugin_name="test.plugin",
            started_at=started,
            ended_at=ended,
        )

        expect_equal(record.duration_s, pytest.approx(1.234, rel=1e-3))
        expect_equal(record.computed_duration_ms, pytest.approx(1234.0, rel=1e-3))


class TestPluginExecutionRecordWithResult:
    """Test PluginExecutionRecord with result field populated."""

    @staticmethod
    def test_result_field_access() -> None:
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

        record_result: PluginResult = expect_is_not_none(record.result)
        rows = expect_is_not_none(record_result.row_counts)
        expect_equal(rows["table_a"], 100)
        expect_equal(rows["table_b"], 200)

    @staticmethod
    def test_result_field_none() -> None:
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

        expect_true(record.result is None)
        expect_equal(record.error, "Something went wrong")

    @staticmethod
    def test_row_counts_via_result() -> None:
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

        row_counts = (
            dict(record.result.row_counts) if record.result and record.result.row_counts else None
        )

        rows = expect_is_not_none(row_counts)
        expect_equal(rows["output.table"], 42)


class TestBasePluginResult:
    """Test BasePluginResult factory methods."""

    @staticmethod
    def test_ok_factory() -> None:
        """Test ok() factory method."""
        result = BasePluginResult.ok(
            row_counts={"table": 10},
            input_hash="abc123",
        )

        expect_true(result.success is True)
        expect_is_not_none(result.row_counts)
        expect_equal(expect_is_not_none(result.row_counts)["table"], 10)
        expect_equal(result.input_hash, "abc123")
        expect_true(result.error is None)
        expect_true(result.skipped is False)

    @staticmethod
    def test_fail_factory() -> None:
        """Test fail() factory method."""
        result = BasePluginResult.fail("Something broke", error_kind="validation")

        expect_true(result.success is False)
        expect_equal(result.error, "Something broke")
        expect_equal(result.error_kind, "validation")
        expect_true(result.row_counts is None)

    @staticmethod
    def test_skip_factory() -> None:
        """Test skip() factory method."""
        result = BasePluginResult.skip("Inputs unchanged")

        expect_true(result.success is True)
        expect_true(result.skipped is True)
        expect_equal(result.skip_reason, "Inputs unchanged")

    @staticmethod
    def test_status_property_succeeded() -> None:
        """Test status property for successful result."""
        result = BasePluginResult.ok()
        expect_equal(result.status, "succeeded")

    @staticmethod
    def test_status_property_failed() -> None:
        """Test status property for failed result."""
        result = BasePluginResult.fail("error")
        expect_equal(result.status, "failed")

    @staticmethod
    def test_status_property_skipped() -> None:
        """Test status property for skipped result."""
        result = BasePluginResult.skip("reason")
        expect_equal(result.status, "skipped")


class TestPluginResult:
    """Test PluginResult factory methods."""

    @staticmethod
    def test_ok_with_artifacts() -> None:
        """Test ok() with artifacts."""
        result = PluginResult.ok(
            row_counts={"table": 5},
            artifacts={"output_file": "/path/to/file"},
        )

        expect_true(result.success is True)
        expect_equal(result.artifacts["output_file"], "/path/to/file")

    @staticmethod
    def test_fail_with_warnings() -> None:
        """Test fail() preserves warnings."""
        result = PluginResult.fail(
            "Critical error",
            warnings=("Warning 1", "Warning 2"),
        )

        expect_true(result.success is False)
        expect_equal(result.error, "Critical error")
        expect_equal(result.warnings, ("Warning 1", "Warning 2"))
