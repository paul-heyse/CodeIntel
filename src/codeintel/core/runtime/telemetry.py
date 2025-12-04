"""Base telemetry infrastructure with OpenTelemetry and Prometheus integration.

This module provides a unified telemetry system that integrates:
- **OpenTelemetry** for distributed tracing (spans, trace context)
- **Prometheus** for metrics (histograms, counters)
- **Structured logging** integration

The implementation gracefully degrades when optional dependencies are
not installed, allowing the core functionality to work without telemetry
in minimal installations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

# Optional OpenTelemetry imports with graceful degradation
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    otel_metrics = None
    otel_trace = None
    StatusCode = None
    OTEL_AVAILABLE = False

# Optional Prometheus imports with graceful degradation
try:
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Histogram as PromHistogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PromCounter = None
    PromHistogram = None
    PROMETHEUS_AVAILABLE = False


log = logging.getLogger(__name__)

# Default histogram buckets following OpenTelemetry HTTP semconv recommendations
DEFAULT_DURATION_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)


@dataclass
class PluginSpan:
    """Unified span for plugin execution tracking.

    Track timing, attributes, and context data for a single plugin
    execution. Used by both OpenTelemetry spans and Prometheus metrics.

    Attributes
    ----------
    plugin_name
        Name of the plugin being executed.
    run_id
        Unique identifier for the execution run.
    start_time_ns
        Start time in nanoseconds (monotonic clock).
    attributes
        Key-value attributes attached to the span.
    context_data
        Internal context data (e.g., OTel span reference).
    """

    plugin_name: str
    run_id: str
    start_time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    context_data: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ns(self) -> int:
        """Return elapsed nanoseconds since span start."""
        return time.perf_counter_ns() - self.start_time_ns

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds since span start."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Return elapsed seconds since span start."""
        return self.elapsed_ns / 1_000_000_000


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for telemetry providers.

    Attributes
    ----------
    service_name
        Service name for OTel resource and Prometheus labels.
    enable_tracing
        Whether to enable OTel tracing.
    enable_metrics
        Whether to enable metrics collection.
    histogram_buckets
        Custom histogram buckets for duration metrics.
    """

    service_name: str = "codeintel"
    enable_tracing: bool = True
    enable_metrics: bool = True
    histogram_buckets: tuple[float, ...] = DEFAULT_DURATION_BUCKETS


class RuntimeTelemetry:
    """Base telemetry manager with OTel and Prometheus integration.

    Provide unified span management and metric recording across plugin
    execution. Gracefully degrade when OTel or Prometheus dependencies
    are not available.

    Parameters
    ----------
    config
        Telemetry configuration. Uses defaults if not provided.

    Examples
    --------
    >>> telemetry = RuntimeTelemetry()
    >>> span = telemetry.start_span("my.plugin", "run-123")
    >>> try:
    ...     # Do work
    ...     telemetry.end_span(span, success=True, rows_written=100)
    ... except Exception as e:
    ...     telemetry.end_span(span, success=False, error=str(e))
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        """Initialize telemetry with optional configuration.

        Parameters
        ----------
        config
            Telemetry configuration. Uses defaults if not provided.
        """
        self._config = config or TelemetryConfig()
        self._tracer: Any | None = None
        self._meter: Any | None = None
        self._otel_duration_histogram: Any | None = None
        self._prom_duration_histogram: Any | None = None
        self._prom_executions_counter: Any | None = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize OTel and Prometheus providers if available."""
        self._init_opentelemetry()
        self._init_prometheus()

    def _init_opentelemetry(self) -> None:
        """Initialize OpenTelemetry tracer and meter."""
        if not OTEL_AVAILABLE:
            log.debug("OpenTelemetry not available; tracing disabled")
            return

        if self._config.enable_tracing and otel_trace is not None:
            self._tracer = otel_trace.get_tracer(self._config.service_name)

        if self._config.enable_metrics and otel_metrics is not None:
            self._meter = otel_metrics.get_meter(self._config.service_name)
            self._otel_duration_histogram = self._meter.create_histogram(
                f"{self._config.service_name}.plugin.duration",
                unit="ms",
                description="Plugin execution duration in milliseconds",
            )

    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            log.debug("Prometheus client not available; metrics disabled")
            return

        if not self._config.enable_metrics:
            return

        # Guard for type narrowing - PromHistogram and PromCounter are guaranteed non-None
        # when PROMETHEUS_AVAILABLE is True
        if PromHistogram is None or PromCounter is None:
            return

        # Sanitize service name for Prometheus metric naming
        metric_prefix = self._config.service_name.replace(".", "_").replace("-", "_")

        # Handle duplicate registration gracefully (can happen in tests)
        try:
            self._prom_duration_histogram = PromHistogram(
                f"{metric_prefix}_plugin_duration_seconds",
                "Plugin execution duration in seconds",
                ["plugin_name", "status"],
                buckets=self._config.histogram_buckets,
            )
        except ValueError:
            # Metric already registered - this is OK in test environments
            log.debug("Prometheus duration histogram already registered for %s", metric_prefix)

        try:
            self._prom_executions_counter = PromCounter(
                f"{metric_prefix}_plugin_executions_total",
                "Total plugin executions",
                ["plugin_name", "status"],
            )
        except ValueError:
            # Metric already registered - this is OK in test environments
            log.debug("Prometheus executions counter already registered for %s", metric_prefix)

    @property
    def service_name(self) -> str:
        """Return the configured service name."""
        return self._config.service_name

    @property
    def config_tracing_enabled(self) -> bool:
        """Return whether tracing is enabled in config."""
        return self._config.enable_tracing

    @property
    def config_metrics_enabled(self) -> bool:
        """Return whether metrics are enabled in config."""
        return self._config.enable_metrics

    @property
    def tracing_enabled(self) -> bool:
        """Check if OTel tracing is available and enabled."""
        return self._tracer is not None

    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is available and enabled."""
        return (
            self._otel_duration_histogram is not None or self._prom_duration_histogram is not None
        )

    def start_span(
        self,
        plugin_name: str,
        run_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> PluginSpan:
        """Start a telemetry span for plugin execution.

        Parameters
        ----------
        plugin_name
            Name of the plugin being executed.
        run_id
            Unique identifier for the execution run.
        attributes
            Optional attributes to attach to the span.

        Returns
        -------
        PluginSpan
            Span object for tracking execution.
        """
        span = PluginSpan(
            plugin_name=plugin_name,
            run_id=run_id,
            start_time_ns=time.perf_counter_ns(),
            attributes=attributes or {},
        )

        # Start OTel span if available
        if self._tracer is not None:
            otel_span = self._tracer.start_span(
                f"plugin.{plugin_name}",
                attributes={
                    "plugin.name": plugin_name,
                    "run.id": run_id,
                    **span.attributes,
                },
            )
            span.context_data["otel_span"] = otel_span

        return span

    def end_span(
        self,
        span: PluginSpan,
        *,
        success: bool,
        rows_written: int = 0,
        error: str | None = None,
    ) -> float:
        """End a telemetry span and record metrics.

        Parameters
        ----------
        span
            Span to end.
        success
            Whether execution succeeded.
        rows_written
            Number of rows written by the plugin.
        error
            Error message if execution failed.

        Returns
        -------
        float
            Duration in seconds.
        """
        duration_s = span.elapsed_s
        duration_ms = span.elapsed_ms
        status: Literal["success", "error"] = "success" if success else "error"

        # Record OTel metrics
        if self._otel_duration_histogram is not None:
            self._otel_duration_histogram.record(
                duration_ms,
                {"plugin": span.plugin_name, "status": status},
            )

        # Record Prometheus metrics
        if self._prom_duration_histogram is not None:
            self._prom_duration_histogram.labels(
                plugin_name=span.plugin_name,
                status=status,
            ).observe(duration_s)

        if self._prom_executions_counter is not None:
            self._prom_executions_counter.labels(
                plugin_name=span.plugin_name,
                status=status,
            ).inc()

        # End OTel span
        otel_span = span.context_data.get("otel_span")
        if otel_span is not None:
            otel_span.set_attribute("duration_ms", duration_ms)
            otel_span.set_attribute("rows_written", rows_written)
            if error:
                otel_span.set_attribute("error", error)
                if StatusCode is not None:
                    otel_span.set_status(StatusCode.ERROR, error)
            elif StatusCode is not None:
                otel_span.set_status(StatusCode.OK)
            otel_span.end()

        # Log completion
        if error:
            log.debug(
                "telemetry.span.end plugin=%s duration=%.3fs success=%s error=%s",
                span.plugin_name,
                duration_s,
                success,
                error,
            )
        else:
            log.debug(
                "telemetry.span.end plugin=%s duration=%.3fs success=%s rows=%d",
                span.plugin_name,
                duration_s,
                success,
                rows_written,
            )

        return duration_s

    @staticmethod
    def record_run_metrics(
        run_id: str,
        *,
        success_count: int,
        failure_count: int,
        skip_count: int,
        duration_s: float,
    ) -> None:
        """Record aggregate metrics for a complete execution run.

        Parameters
        ----------
        run_id
            Unique identifier for the run.
        success_count
            Number of successful plugin executions.
        failure_count
            Number of failed plugin executions.
        skip_count
            Number of skipped plugin executions.
        duration_s
            Total run duration in seconds.
        """
        log.info(
            "telemetry.run.complete run_id=%s success=%d failed=%d skipped=%d duration=%.3fs",
            run_id,
            success_count,
            failure_count,
            skip_count,
            duration_s,
        )


@lru_cache(maxsize=1)
def get_runtime_telemetry() -> RuntimeTelemetry:
    """Return the default runtime telemetry singleton.

    Returns
    -------
    RuntimeTelemetry
        Shared telemetry instance.
    """
    return RuntimeTelemetry()


__all__ = [
    "DEFAULT_DURATION_BUCKETS",
    "OTEL_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "PluginSpan",
    "RuntimeTelemetry",
    "TelemetryConfig",
    "get_runtime_telemetry",
]
