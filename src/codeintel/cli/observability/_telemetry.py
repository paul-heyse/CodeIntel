"""OpenTelemetry integration for CLI observability.

Provide tracing, metrics, and structured logging with automatic
correlation and context propagation.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

from codeintel.core.singleton import SingletonHolder
from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.otel import ObservabilityConfig, bootstrap_observability

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.trace import Span, Tracer

LOG = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry.

    Parameters
    ----------
    enabled
        Whether telemetry is enabled.
    service_name
        Service name for traces.
    export_traces
        Whether to export traces.
    export_metrics
        Whether to export metrics.
    console_export
        Export to console (for debugging).
    otlp_endpoint
        OTLP collector endpoint.
    duckdb_tracing_enabled
        Whether DuckDB tracing spans are enabled.
    duckdb_statement_mode
        Redaction mode for SQL statement spans.
    duckdb_statement_hash_len
        Hash length for redacted statements.
    """

    enabled: bool = True
    service_name: str = "codeintel-cli"
    export_traces: bool = True
    export_metrics: bool = True
    console_export: bool = False
    otlp_endpoint: str | None = None
    duckdb_tracing_enabled: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16

    @classmethod
    def from_settings(
        cls,
        settings: ObservabilitySettings,
        *,
        default_service_name: str,
    ) -> TelemetryConfig:
        """Create config from runtime settings.

        Returns
        -------
        TelemetryConfig
            Configuration derived from runtime settings.
        """
        return cls(
            enabled=settings.enabled,
            service_name=settings.service_name or default_service_name,
            export_traces=settings.export_traces,
            export_metrics=settings.export_metrics,
            console_export=settings.console_export,
            otlp_endpoint=settings.otlp_endpoint,
            duckdb_tracing_enabled=settings.duckdb_tracing_enabled,
            duckdb_statement_mode=settings.duckdb_statement_mode,
            duckdb_statement_hash_len=settings.duckdb_statement_hash_len,
        )


class SpanProtocol(Protocol):
    """Protocol for span-like objects."""

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute on the span."""
        ...

    def record_exception(self, exception: BaseException) -> None:
        """Record an exception on the span."""
        ...

    def end(self) -> None:
        """End the span."""
        ...


def _is_span_like(obj: object) -> bool:
    """Check if an object has span-like interface.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    bool
        True if object has required span methods.
    """
    return (
        callable(getattr(obj, "set_attribute", None))
        and callable(getattr(obj, "record_exception", None))
        and callable(getattr(obj, "end", None))
    )


class _SpanWrapper:
    """Wrapper for span-like objects to provide typed interface."""

    def __init__(self, span: object) -> None:
        """Initialize wrapper.

        Parameters
        ----------
        span
            The underlying span object.
        """
        self._span = span

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute on the span.

        Parameters
        ----------
        key
            Attribute key.
        value
            Attribute value.
        """
        set_attr = getattr(self._span, "set_attribute", None)
        if callable(set_attr):
            set_attr(key, value)

    def record_exception(self, exception: BaseException) -> None:
        """Record an exception on the span.

        Parameters
        ----------
        exception
            Exception to record.
        """
        record_exc = getattr(self._span, "record_exception", None)
        if callable(record_exc):
            record_exc(exception)

    def end(self) -> None:
        """End the span."""
        end_method = getattr(self._span, "end", None)
        if callable(end_method):
            end_method()


def _get_span(obj: object) -> _SpanWrapper:
    """Wrap an object in a typed span interface.

    Parameters
    ----------
    obj
        Object to wrap.

    Returns
    -------
    _SpanWrapper
        Typed wrapper around the span.
    """
    return _SpanWrapper(obj)


class TelemetryProvider:
    """Provider for OpenTelemetry instrumentation.

    Manage tracer and meter instances, providing a facade
    that gracefully degrades when OTEL is not available.

    This class integrates tracing and metrics for CLI operations.
    """

    COMPONENT_NAME: ClassVar[str] = "cli"

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        metrics: OperationMetrics | None = None,
    ) -> None:
        """Initialize the telemetry provider.

        Parameters
        ----------
        config
            Telemetry configuration.
        metrics
            Optional metrics collector (defaults to singleton).
        """
        runtime_settings = load_runtime_settings().observability
        self._config = config or TelemetryConfig.from_settings(
            runtime_settings,
            default_service_name="codeintel-cli",
        )
        self._tracer: Tracer | None = None
        self._initialized = False
        self._metrics = metrics

    def _initialize(self) -> None:
        """Initialize OpenTelemetry via shared bootstrap."""
        if self._initialized or not self._config.enabled:
            return

        runtime = bootstrap_observability(
            ObservabilityConfig(
                enabled=self._config.enabled,
                service_name=self._config.service_name,
                otlp_endpoint=self._config.otlp_endpoint,
                export_traces=self._config.export_traces,
                export_metrics=self._config.export_metrics,
                console_export=self._config.console_export,
                prometheus_enabled=False,
                duckdb_tracing_enabled=self._config.duckdb_tracing_enabled,
                duckdb_statement_mode=self._config.duckdb_statement_mode,
                duckdb_statement_hash_len=self._config.duckdb_statement_hash_len,
            )
        )
        self._tracer = runtime.tracer
        if not runtime.enabled:
            self._config = _dataclass_replace(self._config, enabled=False)
        self._initialized = True
        LOG.debug("OpenTelemetry bootstrap complete")

    @property
    def tracer(self) -> Tracer | None:
        """Get the tracer instance.

        Returns
        -------
        Tracer | None
            Tracer or None if not available.
        """
        self._initialize()
        return self._tracer

    @property
    def metrics(self) -> OperationMetrics:
        """Get the metrics collector.

        Returns
        -------
        OperationMetrics
            Metrics collector instance.
        """
        if self._metrics is None:
            self._metrics = OperationMetricsHolder.get(OperationMetrics)
        return self._metrics

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: dict[str, object] | None = None,
    ) -> Iterator[Span | None]:
        """Create a trace span.

        Parameters
        ----------
        name
            Span name.
        attributes
            Span attributes.

        Yields
        ------
        Span | None
            Active span or None if tracing disabled.
        """
        if not self._config.enabled or self.tracer is None:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span_obj:
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span_obj.set_attribute(key, value)
            yield span_obj


def _dataclass_replace(obj: TelemetryConfig, /, **changes: object) -> TelemetryConfig:
    """Replace fields in a dataclass instance.

    Parameters
    ----------
    obj
        Original dataclass instance.
    **changes
        Fields to replace.

    Returns
    -------
    TelemetryConfig
        New instance with replaced fields.
    """
    enabled = changes.get("enabled", obj.enabled)
    service_name = changes.get("service_name", obj.service_name)
    export_traces = changes.get("export_traces", obj.export_traces)
    export_metrics = changes.get("export_metrics", obj.export_metrics)
    console_export = changes.get("console_export", obj.console_export)
    otlp_endpoint = changes.get("otlp_endpoint", obj.otlp_endpoint)

    return TelemetryConfig(
        enabled=bool(enabled),
        service_name=str(service_name),
        export_traces=bool(export_traces),
        export_metrics=bool(export_metrics),
        console_export=bool(console_export),
        otlp_endpoint=str(otlp_endpoint) if otlp_endpoint is not None else None,
    )


@dataclass
class OperationMetrics:
    """Metrics collector for CLI operations.

    Parameters
    ----------
    operation_counts
        Count of operations by ID and status.
    operation_durations
        Duration histograms by operation ID.
    """

    operation_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    operation_durations: dict[str, list[float]] = field(default_factory=dict)

    def record_operation(
        self,
        operation_id: str,
        *,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Record operation execution.

        Parameters
        ----------
        operation_id
            Operation identifier.
        success
            Whether operation succeeded.
        duration_seconds
            Execution duration.
        """
        status = "success" if success else "error"

        if operation_id not in self.operation_counts:
            self.operation_counts[operation_id] = {"success": 0, "error": 0}
        self.operation_counts[operation_id][status] += 1

        if operation_id not in self.operation_durations:
            self.operation_durations[operation_id] = []
        self.operation_durations[operation_id].append(duration_seconds)

    def get_summary(self) -> dict[str, dict[str, object]]:
        """Get metrics summary.

        Returns
        -------
        dict[str, dict[str, object]]
            Metrics summary.
        """
        summary: dict[str, dict[str, object]] = {}
        for op_id, counts in self.operation_counts.items():
            durations = self.operation_durations.get(op_id, [])
            summary[op_id] = {
                "total_calls": sum(counts.values()),
                "success_count": counts.get("success", 0),
                "error_count": counts.get("error", 0),
                "avg_duration_ms": (sum(durations) / len(durations) * 1000) if durations else 0,
                "p95_duration_ms": self._percentile(durations, 95) * 1000 if durations else 0,
            }
        return summary

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile.

        Parameters
        ----------
        values
            List of values.
        percentile
            Percentile to calculate (0-100).

        Returns
        -------
        float
            Percentile value.
        """
        _ = self
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class TracingMiddleware:
    """Middleware that adds tracing to operations.

    Parameters
    ----------
    provider
        Telemetry provider.
    metrics
        Metrics collector.
    """

    def __init__(
        self,
        provider: TelemetryProvider | None = None,
        metrics: OperationMetrics | None = None,
    ) -> None:
        """Initialize tracing middleware."""
        self._provider = provider or get_telemetry_provider()
        self._metrics = metrics or OperationMetrics()

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        """Start trace span before operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, object]
            Context with span and start time.
        """
        tracer = self._provider.tracer
        span: object = None

        if tracer is not None:
            span = tracer.start_span(
                f"cli.operation.{op_id}",
                attributes={
                    "cli.operation_id": op_id,
                    "cli.param_count": len(params),
                },
            )

        return {
            "span": span,
            "start_time": time.monotonic(),
            "op_id": op_id,
        }

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, object],
    ) -> None:
        """Complete trace span after operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        _ = result

        start_time = context.get("start_time")
        if not isinstance(start_time, float):
            return

        duration = time.monotonic() - start_time
        self._metrics.record_operation(op_id, success=True, duration_seconds=duration)

        span = context.get("span")
        if span is not None and _is_span_like(span):
            span_obj = _get_span(span)
            success_value: bool = True
            span_obj.set_attribute("cli.success", success_value)
            span_obj.set_attribute("cli.duration_ms", duration * 1000)
            span_obj.end()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, object],
    ) -> None:
        """Record error in trace span.

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
        self._metrics.record_operation(op_id, success=False, duration_seconds=duration)

        span = context.get("span")
        if span is not None and _is_span_like(span):
            span_obj = _get_span(span)
            success_value: bool = False
            span_obj.set_attribute("cli.success", success_value)
            span_obj.set_attribute("cli.error_type", type(exc).__name__)
            span_obj.record_exception(exc)
            span_obj.end()


class TelemetryProviderHolder(SingletonHolder[TelemetryProvider]):
    """Thread-safe holder for the shared TelemetryProvider."""


class OperationMetricsHolder(SingletonHolder[OperationMetrics]):
    """Thread-safe holder for the shared OperationMetrics."""


def get_telemetry_provider() -> TelemetryProvider:
    """Get the global telemetry provider.

    Returns
    -------
    TelemetryProvider
        Global provider instance.
    """
    return TelemetryProviderHolder.get(TelemetryProvider)


def get_operation_metrics() -> OperationMetrics:
    """Get the global operation metrics.

    Returns
    -------
    OperationMetrics
        Global metrics instance.
    """
    return OperationMetricsHolder.get(OperationMetrics)


__all__ = [
    "OperationMetrics",
    "TelemetryConfig",
    "TelemetryProvider",
    "TracingMiddleware",
    "get_operation_metrics",
    "get_telemetry_provider",
]
