"""Observability integration for CLI operations.

Provide structured logging configuration for CLI operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

LOG = logging.getLogger(__name__)

# Minimal protocols to describe OpenTelemetry trace module and span context.
@runtime_checkable
class _TraceContext(Protocol):
    trace_id: int
    span_id: int


@runtime_checkable
class _Span(Protocol):
    def get_span_context(self) -> _TraceContext: ...


@runtime_checkable
class _TraceModule(Protocol):
    def get_current_span(self) -> _Span: ...


# Try to import OpenTelemetry trace module (optional dependency)
_otel_trace: _TraceModule | None
try:
    from opentelemetry import trace as _otel_trace

    _OTEL_TRACE_AVAILABLE = True
except ImportError:
    _otel_trace = None
    _OTEL_TRACE_AVAILABLE = False


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
    """

    def __init__(self, *, include_trace: bool = True) -> None:
        """Initialize formatter."""
        super().__init__()
        self._include_trace = include_trace

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

        # Add extra fields from record
        for key in ("operation_id", "duration_ms", "success", "error_type", "params"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add trace context
        if self._include_trace:
            trace_ctx = _get_trace_context()
            if trace_ctx:
                log_data.update(trace_ctx)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def _get_trace_context() -> dict[str, str] | None:
    """Get current trace context.

    Returns
    -------
    dict[str, str] | None
        Trace context or None.
    """
    if not _OTEL_TRACE_AVAILABLE or _otel_trace is None:
        return None

    span = _otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.trace_id:
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return None


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    include_trace: bool = True,
) -> None:
    """Configure structured logging for CLI.

    Parameters
    ----------
    level
        Log level.
    include_trace
        Include trace context.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter(include_trace=include_trace))

    root = logging.getLogger("codeintel.cli")
    root.setLevel(level)
    root.addHandler(handler)


__all__ = [
    "ObservabilityConfig",
    "StructuredLogFormatter",
    "configure_structured_logging",
]
