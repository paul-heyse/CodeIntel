"""Observability integration for CLI operations.

Provide automatic tracing, metrics, and structured logging
for all operations flowing through the executor.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from codeintel.cli.cli_middleware import OperationMiddleware
from codeintel.cli.telemetry import (
    OperationMetrics,
    TelemetryProvider,
    get_operation_metrics,
    get_telemetry_provider,
)

LOG = logging.getLogger(__name__)

# Maximum length for parameter values before truncation
_MAX_PARAM_VALUE_LENGTH = 100

# Try to import OpenTelemetry trace module (optional dependency)
try:
    from opentelemetry import trace as _otel_trace

    _OTEL_TRACE_AVAILABLE = True
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
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


class ObservabilityMiddleware(OperationMiddleware):
    """Middleware that adds comprehensive observability.

    Parameters
    ----------
    config
        Observability configuration.
    telemetry
        Telemetry provider.
    metrics
        Metrics collector.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        telemetry: TelemetryProvider | None = None,
        metrics: OperationMetrics | None = None,
    ) -> None:
        """Initialize middleware."""
        self._config = config or ObservabilityConfig()
        self._telemetry = telemetry or get_telemetry_provider()
        self._metrics = metrics or get_operation_metrics()

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Start observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.
        """
        context: dict[str, Any] = {
            "start_time": time.monotonic(),
            "operation_id": op_id,
        }

        # Start trace span
        if self._config.tracing_enabled:
            span = self._start_span(op_id, params)
            context["span"] = span

        # Log operation start
        if self._config.structured_logging:
            extra: dict[str, Any] = {"operation_id": op_id}
            if self._config.log_params:
                extra["params"] = _sanitize_params(params)
            LOG.info("Operation started", extra=extra)

        return context

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Complete observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        start_time = context.get("start_time")
        if not isinstance(start_time, float):
            return

        duration = time.monotonic() - start_time

        # Record metrics
        if self._config.metrics_enabled:
            self._metrics.record_operation(
                op_id,
                success=True,
                duration_seconds=duration,
            )

        # End trace span
        span = context.get("span")
        if span is not None:
            _end_span(span, success=True, duration=duration)

        # Log completion
        if self._config.structured_logging:
            extra: dict[str, Any] = {
                "operation_id": op_id,
                "duration_ms": duration * 1000,
                "success": True,
            }
            if self._config.log_results and result is not None:
                extra["result_type"] = type(result).__name__
            LOG.info("Operation completed", extra=extra)

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error in observability context.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        start_time = context.get("start_time")
        if not isinstance(start_time, float):
            return

        duration = time.monotonic() - start_time

        # Record metrics
        if self._config.metrics_enabled:
            self._metrics.record_operation(
                op_id,
                success=False,
                duration_seconds=duration,
            )

        # End trace span with error
        span = context.get("span")
        if span is not None:
            _end_span(span, success=False, duration=duration, error=exc)

        # Log error
        if self._config.structured_logging:
            extra: dict[str, Any] = {
                "operation_id": op_id,
                "duration_ms": duration * 1000,
                "success": False,
                "error_type": type(exc).__name__,
            }
            LOG.error("Operation failed", extra=extra, exc_info=exc)

    def _start_span(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> object | None:
        """Start a trace span for operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        object | None
            Started span or None.
        """
        tracer = self._telemetry.tracer
        if tracer is None:
            return None

        return tracer.start_span(
            f"cli.operation.{op_id}",
            attributes={
                "cli.operation_id": op_id,
                "cli.param_count": len(params),
            },
        )


def _end_span(
    span: object,
    *,
    success: bool,
    duration: float,
    error: Exception | None = None,
) -> None:
    """End a span with attributes.

    Parameters
    ----------
    span
        Span to end.
    success
        Whether operation succeeded.
    duration
        Operation duration in seconds.
    error
        Exception if failed.
    """
    set_attr = getattr(span, "set_attribute", None)
    if callable(set_attr):
        set_attr("cli.success", success)
        set_attr("cli.duration_ms", duration * 1000)
        if error:
            set_attr("cli.error_type", type(error).__name__)

    if error:
        record_exc = getattr(span, "record_exception", None)
        if callable(record_exc):
            record_exc(error)

    end_method = getattr(span, "end", None)
    if callable(end_method):
        end_method()


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive data from params for logging.

    Parameters
    ----------
    params
        Parameters to sanitize.

    Returns
    -------
    dict[str, Any]
        Sanitized parameters.
    """
    sensitive_keys = {"password", "token", "secret", "key", "credential", "auth"}
    sanitized: dict[str, Any] = {}
    for key, value in params.items():
        if any(s in key.lower() for s in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LENGTH:
            sanitized[key] = f"{value[:_MAX_PARAM_VALUE_LENGTH]}... (truncated)"
        else:
            sanitized[key] = value
    return sanitized


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


def get_observability_middleware(
    config: ObservabilityConfig | None = None,
) -> ObservabilityMiddleware:
    """Get observability middleware with default configuration.

    Parameters
    ----------
    config
        Optional configuration.

    Returns
    -------
    ObservabilityMiddleware
        Configured middleware.
    """
    return ObservabilityMiddleware(config=config)


__all__ = [
    "ObservabilityConfig",
    "ObservabilityMiddleware",
    "StructuredLogFormatter",
    "configure_structured_logging",
    "get_observability_middleware",
]
