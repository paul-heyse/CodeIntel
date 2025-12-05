"""Coverage tests for ingestion runtime telemetry.

This module provides comprehensive tests for the telemetry infrastructure,
including metric emission, span tracking, and run sink functionality.
Uses real production types per the testing charter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

import pytest

from codeintel.core.execution.telemetry import OTEL_AVAILABLE
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.core.runs import IngestRun, IngestRunMode, IngestRunStatus
from codeintel.ingestion.plugins.protocol import IngestStage
from codeintel.ingestion.runtime.telemetry import (
    IngestRuntimeTelemetry,
    IngestTelemetryConfig,
    OtelIngestRunSink,
    get_ingest_telemetry,
    reset_ingest_telemetry,
)

# =============================================================================
# Real Test Plugin
# =============================================================================


@dataclass
class TelemetryTestPlugin(BaseIngestPlugin):
    """Real plugin for testing telemetry.

    This is a production-style plugin with all required attributes.
    """

    plugin_name: ClassVar[str] = "telemetry_test"
    plugin_description: ClassVar[str] = "Test plugin for telemetry"
    plugin_stage: ClassVar[IngestStage] = "parse"
    produces_tables: ClassVar[tuple[str, ...]] = ("core.test_table", "core.other_table")

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int]:
        """Return test row counts.

        Parameters
        ----------
        ctx
            Execution context (unused in this test plugin).

        Returns
        -------
        Mapping[str, int]
            Row counts for test tables.
        """
        _ = self, ctx
        return {"core.test_table": 10}


# =============================================================================
# IngestTelemetryConfig Tests
# =============================================================================


class TestIngestTelemetryConfig:
    """Tests for IngestTelemetryConfig."""

    def test_default_service_name(self) -> None:
        """Config has correct default service name."""
        config = IngestTelemetryConfig()
        # Default comes from base TelemetryConfig which is "codeintel"
        # IngestTelemetryConfig.service_name is a class attribute hint only
        assert "codeintel" in config.service_name


# =============================================================================
# IngestRuntimeTelemetry Tests
# =============================================================================


class TestIngestRuntimeTelemetry:
    """Tests for IngestRuntimeTelemetry."""

    def test_default_initialization(self) -> None:
        """Telemetry initializes with default settings."""
        telemetry = IngestRuntimeTelemetry()

        assert telemetry is not None

    def test_custom_service_name(self) -> None:
        """Telemetry accepts custom service name."""
        telemetry = IngestRuntimeTelemetry(service_name="custom.service")

        # Should not raise
        assert telemetry is not None

    def test_tracing_disabled(self) -> None:
        """Telemetry works with tracing disabled."""
        telemetry = IngestRuntimeTelemetry(enable_tracing=False)

        # Should not raise
        assert telemetry is not None

    def test_metrics_disabled(self) -> None:
        """Telemetry works with metrics disabled."""
        telemetry = IngestRuntimeTelemetry(enable_metrics=False)

        # Should not raise
        assert telemetry is not None

    def test_both_disabled(self) -> None:
        """Telemetry works with both tracing and metrics disabled."""
        telemetry = IngestRuntimeTelemetry(
            enable_tracing=False,
            enable_metrics=False,
        )

        # Should not raise
        assert telemetry is not None


class TestIngestRuntimeTelemetrySpans:
    """Tests for span tracking functionality."""

    def test_start_plugin_span(self) -> None:
        """Start a plugin span with default attributes."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(plugin, "run-123")

        assert span.plugin_name == "telemetry_test"
        assert span.run_id == "run-123"
        assert span.start_time_ns > 0

    def test_start_plugin_span_with_attributes(self) -> None:
        """Start a plugin span with custom attributes."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(
            plugin,
            "run-456",
            attributes={"custom.attr": "value"},
        )

        assert span.plugin_name == "telemetry_test"
        assert span.run_id == "run-456"

    def test_end_span_success(self) -> None:
        """End a span successfully and get duration."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(plugin, "run-789")
        duration = telemetry.end_span(span, success=True, rows_written=100)

        assert duration >= 0.0

    def test_end_span_failure(self) -> None:
        """End a span with failure and error message."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(plugin, "run-error")
        duration = telemetry.end_span(span, success=False, error="test error")

        assert duration >= 0.0

    def test_end_span_with_rows(self) -> None:
        """End span using end_span_with_rows method."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(plugin, "run-rows")
        duration = telemetry.end_span_with_rows(
            span,
            success=True,
            rows_written=500,
        )

        assert duration >= 0.0

    def test_end_span_with_rows_failure(self) -> None:
        """End span with rows on failure."""
        telemetry = IngestRuntimeTelemetry()
        plugin = TelemetryTestPlugin()

        span = telemetry.start_plugin_span(plugin, "run-fail")
        duration = telemetry.end_span_with_rows(
            span,
            success=False,
            rows_written=0,
            error="failed operation",
        )

        assert duration >= 0.0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestTelemetrySingleton:
    """Tests for telemetry singleton management."""

    def test_get_ingest_telemetry_returns_instance(self) -> None:
        """get_ingest_telemetry returns a telemetry instance."""
        reset_ingest_telemetry()  # Ensure clean state

        telemetry = get_ingest_telemetry()

        assert telemetry is not None
        assert isinstance(telemetry, IngestRuntimeTelemetry)

    def test_get_ingest_telemetry_returns_same_instance(self) -> None:
        """get_ingest_telemetry returns the same singleton instance."""
        reset_ingest_telemetry()  # Ensure clean state

        telemetry1 = get_ingest_telemetry()
        telemetry2 = get_ingest_telemetry()

        assert telemetry1 is telemetry2

    def test_reset_ingest_telemetry(self) -> None:
        """reset_ingest_telemetry creates a new instance on next call."""
        reset_ingest_telemetry()

        telemetry1 = get_ingest_telemetry()
        reset_ingest_telemetry()
        telemetry2 = get_ingest_telemetry()

        # After reset, should be a new instance
        assert telemetry1 is not telemetry2


# =============================================================================
# OtelIngestRunSink Tests (using real IngestRun)
# =============================================================================


def make_test_run(
    *,
    duration_s: float | None = 5.5,
    rows_inserted: int = 100,
    rows_deleted: int = 10,
) -> IngestRun:
    """Create a real IngestRun for testing.

    Parameters
    ----------
    duration_s
        Duration in seconds, or None if not finished.
    rows_inserted
        Number of rows inserted.
    rows_deleted
        Number of rows deleted.

    Returns
    -------
    IngestRun
        A real ingestion run instance.
    """
    now = datetime.now(UTC)
    return IngestRun(
        run_id="test-run-123",
        repo="test/repo",
        commit="abc123",
        step="scip_ingest",
        datasets=("core.modules",),
        mode=IngestRunMode.FULL,
        started_at=now,
        finished_at=now if duration_s is not None else None,
        duration_s=duration_s,
        rows_inserted=rows_inserted,
        rows_deleted=rows_deleted,
        status=IngestRunStatus.OK,
    )


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")
class TestOtelIngestRunSink:
    """Tests for OtelIngestRunSink using real IngestRun."""

    def test_sink_creation_with_otel(self) -> None:
        """OtelIngestRunSink can be created when OTEL is available."""
        sink = OtelIngestRunSink()
        assert sink is not None

    def test_sink_record_with_otel(self) -> None:
        """OtelIngestRunSink.record emits metrics."""
        sink = OtelIngestRunSink()
        run = make_test_run()

        # Should not raise - just verifying it runs
        sink.record(run)

    def test_sink_record_without_duration(self) -> None:
        """OtelIngestRunSink.record handles None duration."""
        sink = OtelIngestRunSink()
        run = make_test_run(duration_s=None, rows_inserted=50, rows_deleted=0)

        # Should not raise
        sink.record(run)

    def test_sink_record_with_error_status(self) -> None:
        """OtelIngestRunSink.record handles error status."""
        sink = OtelIngestRunSink()
        now = datetime.now(UTC)
        run = IngestRun(
            run_id="error-run",
            repo="test/repo",
            commit="abc123",
            step="failing_step",
            datasets=("core.modules",),
            mode=IngestRunMode.INCREMENTAL,
            started_at=now,
            finished_at=now,
            duration_s=1.0,
            rows_inserted=0,
            rows_deleted=0,
            status=IngestRunStatus.ERROR,
            error_kind="ValueError",
            error_message="Test error",
        )

        # Should not raise
        sink.record(run)
