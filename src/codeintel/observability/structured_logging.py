"""Structured logging helpers for observability."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from opentelemetry import trace as otel_trace

if TYPE_CHECKING:
    from structlog.types import Processor

try:
    import orjson
except ImportError:  # pragma: no cover - optional dependency
    orjson = None

try:
    import structlog
except ImportError:  # pragma: no cover - optional dependency
    structlog = None


@runtime_checkable
class TraceAdapter(Protocol):
    """Adapter interface for trace context extraction."""

    def get_trace_context(self) -> dict[str, str] | None:
        """Return trace identifiers for the active span, if any.

        Returns
        -------
        dict[str, str] | None
            Trace identifiers for the active span, if available.
        """
        ...


class OTELTraceAdapter:
    """Trace adapter backed by OpenTelemetry's tracer provider."""

    def __init__(self) -> None:
        self._trace = otel_trace

    def get_trace_context(self) -> dict[str, str] | None:
        """Return trace_id/span_id from the current OpenTelemetry span.

        Returns
        -------
        dict[str, str] | None
            Trace identifiers for the current span, if available.
        """
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


class StructuredLogFormatter(logging.Formatter):
    """Format log records as JSON with optional trace context."""

    def __init__(
        self,
        *,
        include_trace: bool = True,
        trace_adapter: TraceAdapter | None = None,
    ) -> None:
        super().__init__()
        self._include_trace = include_trace
        self._trace_adapter = trace_adapter or get_trace_adapter()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as structured JSON.

        Returns
        -------
        str
            Structured JSON representation of the log record.
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

        if orjson is not None:
            return orjson.dumps(log_data).decode("utf-8")
        return json.dumps(log_data)


def get_trace_adapter() -> TraceAdapter:
    """Return the default trace adapter implementation.

    Returns
    -------
    TraceAdapter
        Trace adapter backed by OpenTelemetry.
    """
    return OTELTraceAdapter()


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    include_trace: bool = True,
    trace_adapter: TraceAdapter | None = None,
) -> None:
    """Configure structured JSON logging for CLI output."""
    _configure_structlog(level=level)
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


def _configure_structlog(*, level: int) -> None:
    if structlog is None:
        return
    serializer = orjson.dumps if orjson is not None else None
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if serializer is not None:
        processors.append(
            cast("Processor", structlog.processors.JSONRenderer(serializer=serializer))
        )
    else:
        processors.append(cast("Processor", structlog.processors.JSONRenderer()))
    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=processors,
        logger_factory=structlog.BytesLoggerFactory(),
    )


__all__ = [
    "OTELTraceAdapter",
    "StructuredLogFormatter",
    "TraceAdapter",
    "configure_structured_logging",
    "get_trace_adapter",
]
