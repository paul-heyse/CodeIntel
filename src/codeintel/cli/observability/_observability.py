"""Observability integration for CLI operations.

Provide structured logging configuration for CLI operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from opentelemetry import trace as otel_trace

LOG = logging.getLogger(__name__)


@runtime_checkable
class TraceAdapter(Protocol):
    """Adapter interface for tracing backends."""

    def get_trace_context(self) -> dict[str, str] | None:
        """Return current trace context identifiers if available.

        Returns
        -------
        dict[str, str] | None
            Mapping with trace/span IDs or None if unavailable.
        """
        ...


class OTELTraceAdapter:
    """Adapter that extracts trace context from OpenTelemetry."""

    def __init__(self) -> None:
        self._trace = otel_trace

    def get_trace_context(self) -> dict[str, str] | None:
        """Return trace_id/span_id from the current span.

        Returns
        -------
        dict[str, str] | None
            Trace identifiers if an active span exists.
        """
        _ = self
        span = self._trace.get_current_span()
        if span is None:
            return None
        span_context = span.get_span_context()
        if not span_context or not getattr(span_context, "trace_id", 0):
            return None
        return {
            "trace_id": format(span_context.trace_id, "032x"),
            "span_id": format(span_context.span_id, "016x"),
        }


@dataclass
class ObservabilityConfig:
    """Configuration for observability features.

    Parameters
    ----------
    tracing_enabled
        Enable trace spans.
    metrics_enabled
        Enable metrics collection.
    structured_logging
        Enable structured log format.
    log_params
        Log operation parameters (privacy consideration).
    log_results
        Log operation results (performance consideration).
    """

    tracing_enabled: bool = True
    metrics_enabled: bool = True
    structured_logging: bool = True
    log_params: bool = False
    log_results: bool = False


class StructuredLogFormatter(logging.Formatter):
    """Log formatter that outputs structured JSON.

    Parameters
    ----------
    include_trace
        Include trace context in logs.
    trace_adapter
        Adapter for extracting trace identifiers.
    """

    def __init__(
        self,
        *,
        include_trace: bool = True,
        trace_adapter: TraceAdapter | None = None,
    ) -> None:
        """Initialize formatter."""
        super().__init__()
        self._include_trace = include_trace
        self._trace_adapter = trace_adapter or get_trace_adapter()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Parameters
        ----------
        record
            Log record.

        Returns
        -------
        str
            JSON formatted log.
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ("operation_id", "duration_ms", "success", "error_type", "params"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        if self._include_trace:
            trace_ctx = self._trace_adapter.get_trace_context()
            if trace_ctx:
                log_data.update(trace_ctx)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_trace_adapter() -> TraceAdapter:
    """Return an active TraceAdapter.

    Returns
    -------
    TraceAdapter
        Adapter backed by OpenTelemetry.
    """
    return OTELTraceAdapter()


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    include_trace: bool = True,
    trace_adapter: TraceAdapter | None = None,
) -> None:
    """Configure structured logging for CLI.

    Parameters
    ----------
    level
        Log level.
    include_trace
        Include trace context.
    trace_adapter
        Optional trace adapter override.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        StructuredLogFormatter(
            include_trace=include_trace,
            trace_adapter=trace_adapter,
        )
    )

    root = logging.getLogger("codeintel.cli")
    root.setLevel(level)
    root.addHandler(handler)


__all__ = [
    "ObservabilityConfig",
    "StructuredLogFormatter",
    "configure_structured_logging",
]
