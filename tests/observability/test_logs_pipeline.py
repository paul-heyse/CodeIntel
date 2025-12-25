"""OpenTelemetry logs pipeline tests."""

from __future__ import annotations

import logging

import pytest

from codeintel.observability.otel import (
    ObservabilityConfig,
    bootstrap_observability,
    shutdown_observability,
)

pytest.importorskip("opentelemetry.sdk._logs")


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_logs_pipeline_bootstrap_adds_handler() -> None:
    """Ensure log handler is attached during bootstrap."""
    shutdown_observability()
    runtime = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            export_logs=True,
            export_traces=False,
            export_metrics=False,
        )
    )
    assert runtime.logger_provider is not None
    assert runtime.log_handler is not None
    assert runtime.log_handler in logging.getLogger().handlers
    shutdown_observability()


def test_logs_pipeline_trace_filter_attached() -> None:
    """Ensure trace filter is attached to log handler when enabled."""
    shutdown_observability()
    runtime = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            export_logs=True,
            export_traces=False,
            export_metrics=False,
            logs_trace_filter=True,
        )
    )
    assert runtime.log_handler is not None
    assert runtime.log_handler.filters
    shutdown_observability()


def test_log_correlation_injects_trace_fields() -> None:
    """Ensure log records include trace correlation fields."""
    shutdown_observability()
    runtime = bootstrap_observability(
        ObservabilityConfig(
            enabled=True,
            service_name="codeintel-test",
            export_logs=False,
            export_traces=False,
            export_metrics=False,
            log_correlation=True,
        )
    )
    handler = _CaptureHandler()
    logger = logging.getLogger("codeintel.logs")
    logger.addHandler(handler)

    if runtime.tracer is not None:
        with runtime.tracer.start_as_current_span("cli.logs"):
            logger.info("log correlation check")

    logger.removeHandler(handler)
    shutdown_observability()
    assert handler.records
    record = handler.records[-1]
    assert getattr(record, "otelTraceID", None)
    assert getattr(record, "otelSpanID", None)
