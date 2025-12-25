"""OpenTelemetry bootstrap and shared runtime access."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.context import Context as ContextType
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as OTLPMetricExporterType,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterType,
    )
    from opentelemetry.metrics import Meter as MeterType
    from opentelemetry.metrics import MeterProvider as MeterProviderApiType
    from opentelemetry.sdk.metrics import MeterProvider as MeterProviderType
    from opentelemetry.sdk.metrics.export import (
        MetricReader,
    )
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader as PeriodicExportingMetricReaderType,
    )
    from opentelemetry.sdk.resources import Resource as ResourceType
    from opentelemetry.sdk.trace import ReadableSpan as ReadableSpanType
    from opentelemetry.sdk.trace import Span as SpanType
    from opentelemetry.sdk.trace import SpanProcessor as SpanProcessorType
    from opentelemetry.sdk.trace import TracerProvider as TracerProviderType
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as BatchSpanProcessorType,
    )
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter as ConsoleSpanExporterType,
    )
    from opentelemetry.trace import Tracer as TracerType
    from opentelemetry.trace import TracerProvider as TracerProviderApiType

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_metrics = None
    otel_trace = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    TracerProvider = None
    Span = None
    ReadableSpan = None
    SpanProcessor = None
    Context = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
except ImportError:
    OTLPMetricExporter = None

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPSpanExporter = None

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    _PROMETHEUS_EXPORTER_AVAILABLE = True
except ImportError:
    _PROMETHEUS_EXPORTER_AVAILABLE = False
    PrometheusMetricReader = None


class _Instrumentor(Protocol):
    """Protocol for OpenTelemetry instrumentors."""

    def instrument(self, **kwargs: object) -> None:
        """Enable instrumentation."""


AsyncioInstrumentor: type[_Instrumentor] | None
try:
    from opentelemetry.instrumentation.asyncio import (
        AsyncioInstrumentor as _AsyncioInstrumentor,
    )

    _ASYNCIO_INSTRUMENTOR_AVAILABLE = True
    AsyncioInstrumentor = _AsyncioInstrumentor
except ImportError:
    _ASYNCIO_INSTRUMENTOR_AVAILABLE = False
    AsyncioInstrumentor = None

HTTPXClientInstrumentor: type[_Instrumentor] | None
try:
    from opentelemetry.instrumentation.httpx import (
        HTTPXClientInstrumentor as _HTTPXClientInstrumentor,
    )

    _HTTPX_INSTRUMENTOR_AVAILABLE = True
    HTTPXClientInstrumentor = _HTTPXClientInstrumentor
except ImportError:
    _HTTPX_INSTRUMENTOR_AVAILABLE = False
    HTTPXClientInstrumentor = None

LoggingInstrumentor: type[_Instrumentor] | None
try:
    from opentelemetry.instrumentation.logging import (
        LoggingInstrumentor as _LoggingInstrumentor,
    )

    _LOGGING_INSTRUMENTOR_AVAILABLE = True
    LoggingInstrumentor = _LoggingInstrumentor
except ImportError:
    _LOGGING_INSTRUMENTOR_AVAILABLE = False
    LoggingInstrumentor = None

RequestsInstrumentor: type[_Instrumentor] | None
try:
    from opentelemetry.instrumentation.requests import (
        RequestsInstrumentor as _RequestsInstrumentor,
    )

    _REQUESTS_INSTRUMENTOR_AVAILABLE = True
    RequestsInstrumentor = _RequestsInstrumentor
except ImportError:
    _REQUESTS_INSTRUMENTOR_AVAILABLE = False
    RequestsInstrumentor = None

ThreadingInstrumentor: type[_Instrumentor] | None
try:
    from opentelemetry.instrumentation.threading import (
        ThreadingInstrumentor as _ThreadingInstrumentor,
    )

    _THREADING_INSTRUMENTOR_AVAILABLE = True
    ThreadingInstrumentor = _ThreadingInstrumentor
except ImportError:
    _THREADING_INSTRUMENTOR_AVAILABLE = False
    ThreadingInstrumentor = None


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Runtime configuration for OpenTelemetry bootstrap."""

    enabled: bool = True
    service_name: str = "codeintel"
    otlp_endpoint: str | None = None
    export_traces: bool = True
    export_metrics: bool = True
    console_export: bool = False
    prometheus_enabled: bool = False
    duckdb_tracing_enabled: bool = True
    duckdb_require_parent_span: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16
    duckdb_query_summary_max_len: int = 255
    duckdb_query_summary_max_targets: int = 6
    duckdb_query_summary_emit_ellipsis: bool = True
    duckdb_query_summary_hash_suspicious_targets: bool = True
    duckdb_query_summary_hash_len: int = 12
    duckdb_query_summary_hash_min_len: int = 64
    duckdb_query_summary_include_subquery_operations: bool = True
    duckdb_query_summary_include_multi_statement: bool = True
    db_query_summary_span_name_hook: bool = False
    duckdb_emit_legacy_db_attributes: bool = False
    duckdb_query_text_policy: str = "never"
    duckdb_query_text_max_len: int = 4096
    duckdb_query_text_strip_comments: bool = True
    duckdb_query_text_collapse_in_lists: bool = True
    duckdb_query_parameter_enabled: bool = False
    duckdb_query_parameter_keys: tuple[str, ...] = ()
    duckdb_query_parameter_hash_keys: tuple[str, ...] = ()
    duckdb_query_parameter_require_in_sql: bool = True
    duckdb_query_parameter_max_str_len: int = 80


@dataclass(frozen=True, slots=True)
class ObservabilityRuntime:
    """Resolved OpenTelemetry runtime handles."""

    enabled: bool
    tracer: TracerType | None
    meter: MeterType | None
    shutdown: Callable[[], ObservabilityShutdownResult] | None
    prometheus_enabled: bool = False
    duckdb_tracing_enabled: bool = True
    duckdb_require_parent_span: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16
    duckdb_query_summary_max_len: int = 255
    duckdb_query_summary_max_targets: int = 6
    duckdb_query_summary_emit_ellipsis: bool = True
    duckdb_query_summary_hash_suspicious_targets: bool = True
    duckdb_query_summary_hash_len: int = 12
    duckdb_query_summary_hash_min_len: int = 64
    duckdb_query_summary_include_subquery_operations: bool = True
    duckdb_query_summary_include_multi_statement: bool = True
    db_query_summary_span_name_hook: bool = False
    duckdb_emit_legacy_db_attributes: bool = False
    duckdb_query_text_policy: str = "never"
    duckdb_query_text_max_len: int = 4096
    duckdb_query_text_strip_comments: bool = True
    duckdb_query_text_collapse_in_lists: bool = True
    duckdb_query_parameter_enabled: bool = False
    duckdb_query_parameter_keys: tuple[str, ...] = ()
    duckdb_query_parameter_hash_keys: tuple[str, ...] = ()
    duckdb_query_parameter_require_in_sql: bool = True
    duckdb_query_parameter_max_str_len: int = 80


class _ObservabilityHolder(SingletonHolder[ObservabilityRuntime]):
    pass


@dataclass(frozen=True, slots=True)
class ObservabilityShutdownResult:
    """Shutdown flush summary for observability runtime."""

    flush_ok: bool
    flush_ms: float
    errors: tuple[str, ...] = ()

    def to_log_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for shutdown flush results.

        Returns
        -------
        dict[str, object]
            Structured payload for logging.
        """
        return {
            "event": "telemetry.flush",
            "telemetry.flush.ok": self.flush_ok,
            "telemetry.flush.duration_ms": self.flush_ms,
            "errors": list(self.errors),
        }


def _force_flush_provider(provider: object, *, label: str, errors: list[str]) -> bool:
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is None:
        return True
    try:
        force_flush()
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        errors.append(f"{label}:{exc}")
        return False
    return True


@dataclass(frozen=True, slots=True)
class _OtelComponents:
    set_meter_provider: Callable[[MeterProviderApiType], None]
    get_meter: Callable[[str], MeterType]
    set_tracer_provider: Callable[[TracerProviderApiType], None]
    get_tracer: Callable[[str], TracerType]
    meter_provider_cls: type[MeterProviderType]
    tracer_provider_cls: type[TracerProviderType]
    periodic_reader_cls: type[PeriodicExportingMetricReaderType]
    batch_span_processor_cls: type[BatchSpanProcessorType]
    console_span_exporter_cls: type[ConsoleSpanExporterType]
    otlp_metric_exporter_cls: type[OTLPMetricExporterType] | None
    otlp_span_exporter_cls: type[OTLPSpanExporterType] | None
    resource_cls: type[ResourceType]


def _package_version() -> str:
    try:
        return version("codeintel")
    except PackageNotFoundError:
        return "unknown"


def _instrument_runtime() -> None:
    if _LOGGING_INSTRUMENTOR_AVAILABLE and LoggingInstrumentor is not None:
        instrumentor = cast("_Instrumentor", LoggingInstrumentor())
        instrumentor.instrument(set_logging_format=False)

    if _THREADING_INSTRUMENTOR_AVAILABLE and ThreadingInstrumentor is not None:
        instrumentor = cast("_Instrumentor", ThreadingInstrumentor())
        instrumentor.instrument()

    if _ASYNCIO_INSTRUMENTOR_AVAILABLE and AsyncioInstrumentor is not None:
        instrumentor = cast("_Instrumentor", AsyncioInstrumentor())
        instrumentor.instrument()

    if _HTTPX_INSTRUMENTOR_AVAILABLE and HTTPXClientInstrumentor is not None:
        instrumentor = cast("_Instrumentor", HTTPXClientInstrumentor())
        instrumentor.instrument()

    if _REQUESTS_INSTRUMENTOR_AVAILABLE and RequestsInstrumentor is not None:
        instrumentor = cast("_Instrumentor", RequestsInstrumentor())
        instrumentor.instrument()


_DbQuerySummarySpanNameProcessor: type[SpanProcessorType] | None

if (
    OTEL_AVAILABLE
    and SpanProcessor is not None
    and Span is not None
    and ReadableSpan is not None
    and Context is not None
):

    class _DbQuerySummarySpanNameProcessorImpl(SpanProcessor):
        _SUMMARY_KEY = "db.query.summary"
        _enabled = True

        def on_start(self, span: SpanType, parent_context: ContextType | None = None) -> None:
            if not self._enabled:
                del parent_context
                return
            attributes = span.attributes
            if attributes is None:
                del parent_context
                return
            summary = attributes.get(self._SUMMARY_KEY)
            if isinstance(summary, str) and summary:
                span.update_name(summary)
            del parent_context

        def on_end(self, span: ReadableSpanType) -> None:
            if not self._enabled:
                del span
                return
            del span

        def shutdown(self) -> None:
            if not self._enabled:
                return

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            if not self._enabled:
                del timeout_millis
                return True
            del timeout_millis
            return True

    _DbQuerySummarySpanNameProcessor = _DbQuerySummarySpanNameProcessorImpl

else:
    _DbQuerySummarySpanNameProcessor = None


def _disabled_runtime() -> ObservabilityRuntime:
    return ObservabilityRuntime(
        enabled=False,
        tracer=None,
        meter=None,
        shutdown=None,
        prometheus_enabled=False,
        duckdb_tracing_enabled=False,
        duckdb_require_parent_span=True,
        duckdb_statement_mode="hash",
        duckdb_statement_hash_len=16,
        duckdb_query_summary_max_len=255,
        duckdb_query_summary_max_targets=6,
        duckdb_query_summary_emit_ellipsis=True,
        duckdb_query_summary_hash_suspicious_targets=True,
        duckdb_query_summary_hash_len=12,
        duckdb_query_summary_hash_min_len=64,
        duckdb_query_summary_include_subquery_operations=True,
        duckdb_query_summary_include_multi_statement=True,
        db_query_summary_span_name_hook=False,
        duckdb_emit_legacy_db_attributes=False,
        duckdb_query_text_policy="never",
        duckdb_query_text_max_len=4096,
        duckdb_query_text_strip_comments=True,
        duckdb_query_text_collapse_in_lists=True,
        duckdb_query_parameter_enabled=False,
        duckdb_query_parameter_keys=(),
        duckdb_query_parameter_hash_keys=(),
        duckdb_query_parameter_require_in_sql=True,
        duckdb_query_parameter_max_str_len=80,
    )


def _resolve_components() -> _OtelComponents | None:
    if not OTEL_AVAILABLE:
        return None

    if otel_metrics is None or otel_trace is None:
        return None

    required = {
        "meter_provider_cls": MeterProvider,
        "tracer_provider_cls": TracerProvider,
        "periodic_reader_cls": PeriodicExportingMetricReader,
        "batch_span_processor_cls": BatchSpanProcessor,
        "console_span_exporter_cls": ConsoleSpanExporter,
        "resource_cls": Resource,
    }
    if any(value is None for value in required.values()):
        return None

    meter_provider_cls = cast("type[MeterProviderType]", required["meter_provider_cls"])
    tracer_provider_cls = cast("type[TracerProviderType]", required["tracer_provider_cls"])
    periodic_reader_cls = cast(
        "type[PeriodicExportingMetricReaderType]",
        required["periodic_reader_cls"],
    )
    batch_span_processor_cls = cast(
        "type[BatchSpanProcessorType]",
        required["batch_span_processor_cls"],
    )
    console_span_exporter_cls = cast(
        "type[ConsoleSpanExporterType]",
        required["console_span_exporter_cls"],
    )
    resource_cls = cast("type[ResourceType]", required["resource_cls"])

    return _OtelComponents(
        set_meter_provider=otel_metrics.set_meter_provider,
        get_meter=otel_metrics.get_meter,
        set_tracer_provider=otel_trace.set_tracer_provider,
        get_tracer=otel_trace.get_tracer,
        meter_provider_cls=meter_provider_cls,
        tracer_provider_cls=tracer_provider_cls,
        periodic_reader_cls=periodic_reader_cls,
        batch_span_processor_cls=batch_span_processor_cls,
        console_span_exporter_cls=console_span_exporter_cls,
        otlp_metric_exporter_cls=OTLPMetricExporter,
        otlp_span_exporter_cls=OTLPSpanExporter,
        resource_cls=resource_cls,
    )


def _build_resource(config: ObservabilityConfig, components: _OtelComponents) -> ResourceType:
    return components.resource_cls.create(
        {
            "service.name": config.service_name,
            "service.version": _package_version(),
        }
    )


def _build_tracer_provider(
    config: ObservabilityConfig,
    resource: ResourceType,
    components: _OtelComponents,
) -> TracerProviderType:
    tracer_provider = components.tracer_provider_cls(resource=resource)
    if not hasattr(tracer_provider, "add_span_processor"):
        LOG.warning("Tracer provider lacks add_span_processor; tracing disabled")
        return tracer_provider
    if config.export_traces and config.otlp_endpoint:
        if components.otlp_span_exporter_cls is None:
            LOG.warning("OTLP span exporter unavailable; trace export disabled")
        else:
            tracer_provider.add_span_processor(
                components.batch_span_processor_cls(
                    components.otlp_span_exporter_cls(endpoint=config.otlp_endpoint)
                )
            )
    if config.console_export:
        tracer_provider.add_span_processor(
            components.batch_span_processor_cls(components.console_span_exporter_cls())
        )
    span_name_processor = _db_query_summary_span_name_processor()
    if config.db_query_summary_span_name_hook and span_name_processor is not None:
        tracer_provider.add_span_processor(span_name_processor)
    return tracer_provider


def _db_query_summary_span_name_processor() -> SpanProcessorType | None:
    if _DbQuerySummarySpanNameProcessor is None:
        return None
    return _DbQuerySummarySpanNameProcessor()


def _build_meter_provider(
    config: ObservabilityConfig,
    resource: ResourceType,
    components: _OtelComponents,
) -> tuple[MeterProviderType, bool]:
    metric_readers: list[MetricReader] = []
    if config.export_metrics and config.otlp_endpoint:
        if components.otlp_metric_exporter_cls is None:
            LOG.warning("OTLP metric exporter unavailable; metrics export disabled")
        else:
            metric_readers.append(
                components.periodic_reader_cls(
                    components.otlp_metric_exporter_cls(endpoint=config.otlp_endpoint)
                )
            )

    prometheus_enabled = False
    if config.prometheus_enabled:
        if _PROMETHEUS_EXPORTER_AVAILABLE and PrometheusMetricReader is not None:
            metric_readers.append(PrometheusMetricReader())
            prometheus_enabled = True
        else:
            LOG.warning("Prometheus exporter unavailable; /metrics will be disabled")

    meter_provider = components.meter_provider_cls(
        resource=resource,
        metric_readers=metric_readers,
    )
    return meter_provider, prometheus_enabled


def _build_shutdown(
    tracer_provider: TracerProviderType,
    meter_provider: MeterProviderType,
) -> Callable[[], ObservabilityShutdownResult]:
    def _shutdown() -> ObservabilityShutdownResult:
        start = time.perf_counter()
        errors: list[str] = []
        flush_ok = True
        try:
            tracer_provider.shutdown()
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            flush_ok = False
            errors.append(f"tracer:{exc}")
        try:
            meter_provider.shutdown()
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            flush_ok = False
            errors.append(f"meter:{exc}")
        duration_ms = (time.perf_counter() - start) * 1000
        return ObservabilityShutdownResult(
            flush_ok=flush_ok,
            flush_ms=duration_ms,
            errors=tuple(errors),
        )

    return _shutdown


def _init_observability(config: ObservabilityConfig) -> ObservabilityRuntime:
    if not config.enabled:
        return _disabled_runtime()

    components = _resolve_components()
    if components is None:
        return _disabled_runtime()

    _instrument_runtime()

    resource = _build_resource(config, components)
    tracer_provider = _build_tracer_provider(config, resource, components)
    components.set_tracer_provider(tracer_provider)
    tracer = components.get_tracer(config.service_name)

    meter_provider, prometheus_enabled = _build_meter_provider(config, resource, components)
    components.set_meter_provider(meter_provider)
    meter = components.get_meter(config.service_name)

    shutdown = _build_shutdown(tracer_provider, meter_provider)
    return ObservabilityRuntime(
        enabled=True,
        tracer=tracer,
        meter=meter,
        shutdown=shutdown,
        prometheus_enabled=prometheus_enabled,
        duckdb_tracing_enabled=config.duckdb_tracing_enabled,
        duckdb_require_parent_span=config.duckdb_require_parent_span,
        duckdb_statement_mode=config.duckdb_statement_mode,
        duckdb_statement_hash_len=config.duckdb_statement_hash_len,
        duckdb_query_summary_max_len=config.duckdb_query_summary_max_len,
        duckdb_query_summary_max_targets=config.duckdb_query_summary_max_targets,
        duckdb_query_summary_emit_ellipsis=config.duckdb_query_summary_emit_ellipsis,
        duckdb_query_summary_hash_suspicious_targets=(
            config.duckdb_query_summary_hash_suspicious_targets
        ),
        duckdb_query_summary_hash_len=config.duckdb_query_summary_hash_len,
        duckdb_query_summary_hash_min_len=config.duckdb_query_summary_hash_min_len,
        duckdb_query_summary_include_subquery_operations=(
            config.duckdb_query_summary_include_subquery_operations
        ),
        duckdb_query_summary_include_multi_statement=(
            config.duckdb_query_summary_include_multi_statement
        ),
        db_query_summary_span_name_hook=config.db_query_summary_span_name_hook,
        duckdb_emit_legacy_db_attributes=config.duckdb_emit_legacy_db_attributes,
        duckdb_query_text_policy=config.duckdb_query_text_policy,
        duckdb_query_text_max_len=config.duckdb_query_text_max_len,
        duckdb_query_text_strip_comments=config.duckdb_query_text_strip_comments,
        duckdb_query_text_collapse_in_lists=config.duckdb_query_text_collapse_in_lists,
        duckdb_query_parameter_enabled=config.duckdb_query_parameter_enabled,
        duckdb_query_parameter_keys=config.duckdb_query_parameter_keys,
        duckdb_query_parameter_hash_keys=config.duckdb_query_parameter_hash_keys,
        duckdb_query_parameter_require_in_sql=config.duckdb_query_parameter_require_in_sql,
        duckdb_query_parameter_max_str_len=config.duckdb_query_parameter_max_str_len,
    )


def bootstrap_observability(config: ObservabilityConfig) -> ObservabilityRuntime:
    """Initialize OpenTelemetry providers (idempotent).

    Returns
    -------
    ObservabilityRuntime
        Initialized observability runtime handles.
    """
    return _ObservabilityHolder.get(lambda: _init_observability(config))


def get_observability() -> ObservabilityRuntime:
    """Return the active observability runtime state.

    Returns
    -------
    ObservabilityRuntime
        Active runtime handles or a disabled runtime when uninitialized.
    """
    runtime = _ObservabilityHolder.get_or_none()
    if runtime is not None:
        return runtime
    return ObservabilityRuntime(
        enabled=False,
        tracer=None,
        meter=None,
        shutdown=None,
        prometheus_enabled=False,
        duckdb_tracing_enabled=False,
        duckdb_require_parent_span=True,
        duckdb_statement_mode="hash",
        duckdb_statement_hash_len=16,
        duckdb_query_summary_max_len=255,
        duckdb_query_summary_max_targets=6,
        duckdb_query_summary_emit_ellipsis=True,
        duckdb_query_summary_hash_suspicious_targets=True,
        duckdb_query_summary_hash_len=12,
        duckdb_query_summary_hash_min_len=64,
        duckdb_query_summary_include_subquery_operations=True,
        duckdb_query_summary_include_multi_statement=True,
        db_query_summary_span_name_hook=False,
        duckdb_query_text_policy="never",
        duckdb_query_text_max_len=4096,
        duckdb_query_text_strip_comments=True,
        duckdb_query_text_collapse_in_lists=True,
        duckdb_query_parameter_enabled=False,
        duckdb_query_parameter_keys=(),
        duckdb_query_parameter_hash_keys=(),
        duckdb_query_parameter_require_in_sql=True,
        duckdb_query_parameter_max_str_len=80,
    )


def shutdown_observability() -> ObservabilityShutdownResult | None:
    """Shut down the active observability runtime, if available.

    Returns
    -------
    ObservabilityShutdownResult | None
        Structured flush results, or None if observability is inactive.
    """
    runtime = _ObservabilityHolder.get_or_none()
    if runtime is None or runtime.shutdown is None:
        return None
    result: ObservabilityShutdownResult | None = None
    try:
        result = runtime.shutdown()
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        result = ObservabilityShutdownResult(
            flush_ok=False,
            flush_ms=0.0,
            errors=(str(exc),),
        )
    if result is not None:
        _log_shutdown_result(result)
    _ObservabilityHolder.reset()
    return result


def flush_observability() -> ObservabilityShutdownResult | None:
    """Force-flush the active observability runtime without shutting it down.

    Returns
    -------
    ObservabilityShutdownResult | None
        Structured flush results, or None if observability is inactive.
    """
    runtime = _ObservabilityHolder.get_or_none()
    if runtime is None or not runtime.enabled:
        return None
    if otel_trace is None or otel_metrics is None:
        return None
    start = time.perf_counter()
    errors: list[str] = []
    flush_ok = True
    tracer_provider = otel_trace.get_tracer_provider()
    meter_provider = otel_metrics.get_meter_provider()
    if tracer_provider is not None:
        flush_ok = _force_flush_provider(
            tracer_provider,
            label="tracer",
            errors=errors,
        ) and flush_ok
    if meter_provider is not None:
        flush_ok = _force_flush_provider(
            meter_provider,
            label="meter",
            errors=errors,
        ) and flush_ok
    duration_ms = (time.perf_counter() - start) * 1000
    return ObservabilityShutdownResult(
        flush_ok=flush_ok,
        flush_ms=duration_ms,
        errors=tuple(errors),
    )


def _log_shutdown_result(result: ObservabilityShutdownResult) -> None:
    payload = json.dumps(result.to_log_payload(), sort_keys=True)
    if result.flush_ok:
        LOG.info("telemetry.flush %s", payload)
    else:
        LOG.warning("telemetry.flush %s", payload)


__all__ = [
    "ObservabilityConfig",
    "ObservabilityRuntime",
    "ObservabilityShutdownResult",
    "bootstrap_observability",
    "flush_observability",
    "get_observability",
    "observability_config_from_settings",
    "shutdown_observability",
]


def observability_config_from_settings(
    settings: ObservabilitySettings,
    *,
    default_service_name: str,
) -> ObservabilityConfig:
    """Build observability configuration from settings.

    Returns
    -------
    ObservabilityConfig
        Configuration derived from runtime settings.
    """
    service_name = settings.service_name or default_service_name
    return ObservabilityConfig(
        enabled=settings.enabled,
        service_name=service_name,
        otlp_endpoint=settings.otlp_endpoint,
        export_traces=settings.export_traces,
        export_metrics=settings.export_metrics,
        console_export=settings.console_export,
        prometheus_enabled=settings.prometheus_enabled,
        duckdb_tracing_enabled=settings.duckdb_tracing_enabled,
        duckdb_require_parent_span=settings.duckdb_require_parent_span,
        duckdb_statement_mode=settings.duckdb_statement_mode,
        duckdb_statement_hash_len=settings.duckdb_statement_hash_len,
        duckdb_query_summary_max_len=settings.duckdb_query_summary_max_len,
        duckdb_query_summary_max_targets=settings.duckdb_query_summary_max_targets,
        duckdb_query_summary_emit_ellipsis=settings.duckdb_query_summary_emit_ellipsis,
        duckdb_query_summary_hash_suspicious_targets=(
            settings.duckdb_query_summary_hash_suspicious_targets
        ),
        duckdb_query_summary_hash_len=settings.duckdb_query_summary_hash_len,
        duckdb_query_summary_hash_min_len=settings.duckdb_query_summary_hash_min_len,
        duckdb_query_summary_include_subquery_operations=(
            settings.duckdb_query_summary_include_subquery_operations
        ),
        duckdb_query_summary_include_multi_statement=(
            settings.duckdb_query_summary_include_multi_statement
        ),
        db_query_summary_span_name_hook=settings.db_query_summary_span_name_hook,
        duckdb_emit_legacy_db_attributes=settings.duckdb_emit_legacy_db_attributes,
        duckdb_query_text_policy=settings.duckdb_query_text_policy,
        duckdb_query_text_max_len=settings.duckdb_query_text_max_len,
        duckdb_query_text_strip_comments=settings.duckdb_query_text_strip_comments,
        duckdb_query_text_collapse_in_lists=settings.duckdb_query_text_collapse_in_lists,
        duckdb_query_parameter_enabled=settings.duckdb_query_parameter_enabled,
        duckdb_query_parameter_keys=settings.duckdb_query_parameter_keys,
        duckdb_query_parameter_hash_keys=settings.duckdb_query_parameter_hash_keys,
        duckdb_query_parameter_require_in_sql=settings.duckdb_query_parameter_require_in_sql,
        duckdb_query_parameter_max_str_len=settings.duckdb_query_parameter_max_str_len,
    )
