"""OpenTelemetry bootstrap and shared runtime access."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.core.config.settings import ObservabilitySettings
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from collections.abc import Callable

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
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_metrics = None
    otel_trace = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None

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
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16


@dataclass(frozen=True, slots=True)
class ObservabilityRuntime:
    """Resolved OpenTelemetry runtime handles."""

    enabled: bool
    tracer: TracerType | None
    meter: MeterType | None
    shutdown: Callable[[], None] | None
    prometheus_enabled: bool = False
    duckdb_tracing_enabled: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16


class _ObservabilityHolder(SingletonHolder[ObservabilityRuntime]):
    pass


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
    otlp_metric_exporter_cls: type[OTLPMetricExporterType]
    otlp_span_exporter_cls: type[OTLPSpanExporterType]
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



def _disabled_runtime() -> ObservabilityRuntime:
    return ObservabilityRuntime(
        enabled=False,
        tracer=None,
        meter=None,
        shutdown=None,
        prometheus_enabled=False,
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
        "otlp_metric_exporter_cls": OTLPMetricExporter,
        "otlp_span_exporter_cls": OTLPSpanExporter,
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
    otlp_metric_exporter_cls = cast(
        "type[OTLPMetricExporterType]",
        required["otlp_metric_exporter_cls"],
    )
    otlp_span_exporter_cls = cast(
        "type[OTLPSpanExporterType]",
        required["otlp_span_exporter_cls"],
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
        otlp_metric_exporter_cls=otlp_metric_exporter_cls,
        otlp_span_exporter_cls=otlp_span_exporter_cls,
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
    if config.export_traces and config.otlp_endpoint:
        tracer_provider.add_span_processor(
            components.batch_span_processor_cls(
                components.otlp_span_exporter_cls(endpoint=config.otlp_endpoint)
            )
        )
    if config.console_export:
        tracer_provider.add_span_processor(
            components.batch_span_processor_cls(components.console_span_exporter_cls())
        )
    return tracer_provider


def _build_meter_provider(
    config: ObservabilityConfig,
    resource: ResourceType,
    components: _OtelComponents,
) -> tuple[MeterProviderType, bool]:
    metric_readers: list[MetricReader] = []
    if config.export_metrics and config.otlp_endpoint:
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
) -> Callable[[], None]:
    def _shutdown() -> None:
        try:
            tracer_provider.shutdown()
        except RuntimeError:
            LOG.debug("Tracer provider shutdown skipped")
        try:
            meter_provider.shutdown()
        except RuntimeError:
            LOG.debug("Meter provider shutdown skipped")

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
        duckdb_statement_mode=config.duckdb_statement_mode,
        duckdb_statement_hash_len=config.duckdb_statement_hash_len,
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
        duckdb_statement_mode="hash",
        duckdb_statement_hash_len=16,
    )


def shutdown_observability() -> None:
    """Shut down the active observability runtime, if available."""
    runtime = _ObservabilityHolder.get_or_none()
    if runtime is None or runtime.shutdown is None:
        return
    runtime.shutdown()
    _ObservabilityHolder.reset()


__all__ = [
    "ObservabilityConfig",
    "ObservabilityRuntime",
    "bootstrap_observability",
    "get_observability",
    "observability_config_from_settings",
    "shutdown_observability",
]


def observability_config_from_settings(
    settings: ObservabilitySettings,
    *,
    default_service_name: str,
) -> ObservabilityConfig:
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
        duckdb_statement_mode=settings.duckdb_statement_mode,
        duckdb_statement_hash_len=settings.duckdb_statement_hash_len,
    )
