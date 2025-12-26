"""OpenTelemetry runtime bootstrap and shared access."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from weakref import WeakKeyDictionary

import grpc
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as OTLPMetricExporterGrpc,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGrpc,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as OTLPMetricExporterHttp,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterHttp,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.metrics import (
    AlwaysOffExemplarFilter,
    AlwaysOnExemplarFilter,
    MeterProvider,
    TraceBasedExemplarFilter,
)
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from codeintel.core.config.settings import (
    BatchProcessorSettings,
    GrpcObservabilitySettings,
    HamiltonTrackerSettings,
    LogLimitSettings,
    MetricExportSettings,
    MetricViewSettings,
    ObservabilitySettings,
    OtlpExporterSettings,
    SpanLimitSettings,
)
from codeintel.core.singleton import SingletonHolder
from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.config_loader import apply_otel_config_file, load_otel_config_file
from codeintel.observability.events import TelemetryEvent, emit_event
from codeintel.observability.grpc import GrpcObservabilityHandle, register_grpc_observability
from codeintel.observability.instrumentation_registry import (
    InstrumentationRegistry,
    get_instrumentation_registry,
)
from codeintel.observability.policy import ObservabilityPolicy, policy_from_settings
from codeintel.observability.semconv_keys import (
    DB_QUERY_SUMMARY,
    TELEMETRY_ACTION,
    TELEMETRY_FLUSH_MS,
    TELEMETRY_FLUSH_OK,
)
from codeintel.observability.test_mode import TestTelemetryMode, resolve_test_telemetry_mode

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.sdk.metrics import ExemplarFilter

LOG = logging.getLogger(__name__)
DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0


class _LoggerProvider(Protocol):
    """Protocol for logger provider behavior used by shutdown hooks."""

    def add_log_record_processor(self, processor: object) -> None:
        """Register a log record processor."""
        ...

    def shutdown(self) -> None:
        """Shutdown the provider."""
        ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending log records."""
        ...


class _DataclassFields(Protocol):
    """Protocol for dataclass instances exposing field metadata."""

    __dataclass_fields__: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """Resource metadata for OpenTelemetry spans and metrics."""

    service_name: str = "codeintel"
    service_version: str | None = None
    deployment_environment: str | None = None
    resource_attributes: tuple[tuple[str, str], ...] = ()
    propagators: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExporterConfig:
    """Exporter settings for OpenTelemetry signals."""

    otlp: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_traces: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_metrics: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_logs: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Tracing configuration for OpenTelemetry."""

    enabled: bool = True
    sampler: str | None = None
    sampler_arg: float | None = None
    batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    span_limits: SpanLimitSettings = field(default_factory=SpanLimitSettings)
    console_export: bool = False


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Metric configuration for OpenTelemetry."""

    enabled: bool = True
    export: MetricExportSettings = field(default_factory=MetricExportSettings)
    exemplar_filter: str | None = None
    views: MetricViewSettings = field(default_factory=MetricViewSettings)
    prometheus_enabled: bool = False


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Log configuration for OpenTelemetry."""

    enabled: bool = False
    auto_instrument: bool = False
    correlation: bool = False
    trace_filter: bool = False
    batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    limits: LogLimitSettings = field(default_factory=LogLimitSettings)


@dataclass(frozen=True, slots=True)
class LoggingPipelineState:
    """Resolved logger provider and handler for the logging pipeline."""

    logger_provider: _LoggerProvider | None
    log_handler: logging.Handler | None


@dataclass(slots=True)
class LoggingPipeline:
    """Own the structured logging pipeline for observability."""

    config: ObservabilityConfig
    resource: Resource
    registry: InstrumentationRegistry

    def install(self, *, force_handler: bool = False) -> LoggingPipelineState:
        """Initialize logging providers/handlers and instrumentation.

        Returns
        -------
        LoggingPipelineState
            Installed logging pipeline state.
        """
        logger_provider, log_handler = _build_logger_provider_with_handler(
            self.config,
            self.resource,
            force_handler=force_handler,
        )
        _instrument_logging(self.config, self.registry)
        return LoggingPipelineState(
            logger_provider=logger_provider,
            log_handler=log_handler,
        )


@dataclass(frozen=True, slots=True)
class CliTelemetryConfig:
    """CLI telemetry configuration."""

    enabled: bool = True
    args_allowlist: tuple[str, ...] = ()
    args_capture_mode: str = "names-only"


@dataclass(frozen=True, slots=True)
class TeardownConfig:
    """Shutdown telemetry configuration."""

    enabled: bool = True
    task_sample_limit: int = 5
    thread_sample_limit: int = 5
    subprocess_sample_limit: int = 5


@dataclass(frozen=True, slots=True)
class DbTracingConfig:
    """Database tracing configuration for DuckDB."""

    enabled: bool = True
    require_parent_span: bool = True
    statement_mode: str = "hash"
    statement_hash_len: int = 16
    query_summary_max_len: int = 255
    query_summary_max_targets: int = 6
    query_summary_emit_ellipsis: bool = True
    query_summary_hash_suspicious_targets: bool = True
    query_summary_hash_len: int = 12
    query_summary_hash_min_len: int = 64
    query_summary_include_subquery_operations: bool = True
    query_summary_include_multi_statement: bool = True
    query_summary_span_name_hook: bool = False
    query_text_policy: str = "never"
    query_text_max_len: int = 4096
    query_text_strip_comments: bool = True
    query_text_collapse_in_lists: bool = True
    query_parameter_enabled: bool = False
    query_parameter_keys: tuple[str, ...] = ()
    query_parameter_hash_keys: tuple[str, ...] = ()
    query_parameter_require_in_sql: bool = True
    query_parameter_max_str_len: int = 80


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Runtime configuration for OpenTelemetry bootstrap."""

    enabled: bool = True
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    exporters: ExporterConfig = field(default_factory=ExporterConfig)
    traces: TraceConfig = field(default_factory=TraceConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    cli: CliTelemetryConfig = field(default_factory=CliTelemetryConfig)
    teardown: TeardownConfig = field(default_factory=TeardownConfig)
    grpc_observability: GrpcObservabilitySettings = field(default_factory=GrpcObservabilitySettings)
    hamilton_tracker: HamiltonTrackerSettings = field(default_factory=HamiltonTrackerSettings)
    db_tracing: DbTracingConfig = field(default_factory=DbTracingConfig)
    config_file: Path | None = None
    test_mode: TestTelemetryMode | None = None
    policy: ObservabilityPolicy = field(default_factory=ObservabilityPolicy)


class ConfigSource(StrEnum):
    """Source of a resolved configuration value."""

    DEFAULT = "default"
    SETTINGS = "settings"
    OVERRIDE = "override"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConfigProvenance:
    """Provenance mapping for resolved configuration fields."""

    sources: Mapping[str, ConfigSource]

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-serializable provenance payload.

        Returns
        -------
        dict[str, str]
            Mapping of config field path to provenance source.
        """
        return {key: source.value for key, source in self.sources.items()}

    @classmethod
    def unknown_for(cls, config: ObservabilityConfig) -> ConfigProvenance:
        """Return provenance with unknown sources for every field.

        Returns
        -------
        ConfigProvenance
            Provenance with unknown sources for every field.
        """
        flattened = _flatten_config(config)
        return cls(dict.fromkeys(flattened, ConfigSource.UNKNOWN))


@dataclass(frozen=True, slots=True)
class ResolvedObservabilityConfig:
    """Resolved observability configuration with provenance metadata."""

    config: ObservabilityConfig
    provenance: ConfigProvenance

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of config + provenance.

        Returns
        -------
        dict[str, object]
            Serialized config and provenance payload.
        """
        return {
            "config": _serialize_config(self.config),
            "provenance": self.provenance.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class ConfigResolver:
    """Resolve observability configuration with provenance tracking."""

    default_service_name: str

    def resolve(
        self,
        settings: ObservabilitySettings,
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> ResolvedObservabilityConfig:
        """Resolve an observability config from settings and overrides.

        Returns
        -------
        ResolvedObservabilityConfig
            Resolved config with provenance metadata.
        """
        config = _observability_config_from_settings(
            settings,
            default_service_name=self.default_service_name,
        )
        sources = dict.fromkeys(_flatten_config(config), ConfigSource.SETTINGS)
        if not settings.service_name:
            sources["resources.service_name"] = ConfigSource.DEFAULT
        if overrides:
            config = _apply_overrides(config, overrides)
            for key in overrides:
                sources[key] = ConfigSource.OVERRIDE
        return ResolvedObservabilityConfig(
            config=config,
            provenance=ConfigProvenance(sources),
        )


@dataclass(frozen=True, slots=True)
class TestTelemetryHandles:
    """In-memory telemetry handles for test assertions."""

    span_exporter: InMemorySpanExporter | None
    metric_reader: InMemoryMetricReader | None


@dataclass(frozen=True, slots=True)
class ObservabilityRuntime:
    """Resolved OpenTelemetry runtime handles."""

    enabled: bool
    tracer: Tracer | None
    meter: Meter | None
    logger_provider: _LoggerProvider | None
    log_handler: logging.Handler | None
    shutdown: Callable[[], ObservabilityShutdownResult] | None
    policy: ObservabilityPolicy
    prometheus_enabled: bool = False
    grpc_observability: GrpcObservabilityHandle | None = None
    db_tracing: DbTracingConfig = field(default_factory=DbTracingConfig)
    test_handles: TestTelemetryHandles | None = None
    config: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    config_provenance: ConfigProvenance | None = None


class _ObservabilityHolder(SingletonHolder[ObservabilityRuntime]):
    pass


@dataclass(slots=True)
class PipelineHealthState:
    """Snapshot of last telemetry pipeline flush attempt."""

    last_flush_ok: bool | None = None
    last_flush_ms: float | None = None
    last_flush_errors: tuple[str, ...] = ()


class ObservabilityRuntimeManager:
    """Lifecycle manager for observability runtime state."""

    @staticmethod
    def bootstrap(
        config: ObservabilityConfig | ResolvedObservabilityConfig,
    ) -> ObservabilityRuntime:
        """Initialize and return the observability runtime.

        Returns
        -------
        ObservabilityRuntime
            Initialized observability runtime.
        """
        resolved = _coerce_resolved_config(config)
        return _ObservabilityHolder.get(lambda: _init_observability(resolved))

    @staticmethod
    def get() -> ObservabilityRuntime:
        """Return the current runtime, or a disabled runtime.

        Returns
        -------
        ObservabilityRuntime
            Active runtime or a disabled runtime when uninitialized.
        """
        runtime = _ObservabilityHolder.get_or_none()
        if runtime is not None:
            return runtime
        return _disabled_runtime()

    @staticmethod
    def shutdown() -> ObservabilityShutdownResult | None:
        """Shut down the runtime and reset state.

        Returns
        -------
        ObservabilityShutdownResult | None
            Flush summary for the runtime or None when inactive.
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
            _log_shutdown_result(result, policy=runtime.policy)
            _record_pipeline_health(result)
            _record_pipeline_metrics(result, action="shutdown")
        _ObservabilityHolder.reset()
        return result

    @staticmethod
    def flush() -> ObservabilityShutdownResult | None:
        """Force-flush the runtime without shutdown.

        Returns
        -------
        ObservabilityShutdownResult | None
            Flush summary for the runtime or None when inactive.
        """
        runtime = _ObservabilityHolder.get_or_none()
        if runtime is None or not runtime.enabled:
            return None
        start = time.perf_counter()
        errors: list[str] = []
        flush_ok = True

        tracer_provider = otel_trace.get_tracer_provider()
        meter_provider = otel_metrics.get_meter_provider()

        if tracer_provider is not None:
            flush_ok = (
                _force_flush_provider(tracer_provider, label="tracer", errors=errors) and flush_ok
            )
        if meter_provider is not None:
            flush_ok = (
                _force_flush_provider(meter_provider, label="meter", errors=errors) and flush_ok
            )
        if runtime.logger_provider is not None:
            flush_ok = (
                _force_flush_provider(runtime.logger_provider, label="logger", errors=errors)
                and flush_ok
            )

        duration_ms = (time.perf_counter() - start) * 1000
        result = ObservabilityShutdownResult(
            flush_ok=flush_ok,
            flush_ms=duration_ms,
            errors=tuple(errors),
        )
        _record_pipeline_health(result)
        _record_pipeline_metrics(result, action="flush")
        return result

    @staticmethod
    def reset() -> None:
        """Clear any cached runtime state."""
        _ObservabilityHolder.reset()


_RUNTIME_MANAGER = ObservabilityRuntimeManager()
_PIPELINE_HEALTH_STATE = PipelineHealthState()


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
            TELEMETRY_FLUSH_OK: self.flush_ok,
            TELEMETRY_FLUSH_MS: self.flush_ms,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class _ResolvedOtlp:
    endpoint: str | None
    protocol: str
    headers: tuple[tuple[str, str], ...]
    timeout_s: float | None
    compression: str | None
    certificate: str | None
    client_certificate: str | None
    client_key: str | None
    insecure: bool | None


class _DbQuerySummarySpanNameProcessor(SpanProcessor):
    _SUMMARY_KEY = DB_QUERY_SUMMARY
    _enabled = True

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
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

    def on_end(self, span: ReadableSpan) -> None:
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


def _db_query_summary_span_name_processor() -> SpanProcessor | None:
    return _DbQuerySummarySpanNameProcessor()


def _package_version() -> str:
    try:
        return version("codeintel")
    except PackageNotFoundError:
        return "unknown"


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


def _resolve_otlp(base: OtlpExporterSettings, override: OtlpExporterSettings) -> _ResolvedOtlp:
    headers = override.headers or base.headers
    protocol = (override.protocol or base.protocol or "grpc").strip().lower()
    return _ResolvedOtlp(
        endpoint=override.endpoint or base.endpoint,
        protocol=protocol,
        headers=headers,
        timeout_s=override.timeout_s if override.timeout_s is not None else base.timeout_s,
        compression=override.compression or base.compression,
        certificate=override.certificate or base.certificate,
        client_certificate=override.client_certificate or base.client_certificate,
        client_key=override.client_key or base.client_key,
        insecure=override.insecure if override.insecure is not None else base.insecure,
    )


def _resolve_resource_attributes(config: ObservabilityConfig) -> dict[str, str]:
    attrs = dict(config.resources.resource_attributes)
    attrs.setdefault("service.name", config.resources.service_name)
    service_version = config.resources.service_version or _package_version()
    if service_version:
        attrs.setdefault("service.version", service_version)
    if config.resources.deployment_environment:
        attrs.setdefault("deployment.environment.name", config.resources.deployment_environment)
    return attrs


def _build_resource(config: ObservabilityConfig) -> Resource:
    attrs = _resolve_resource_attributes(config)
    return Resource.create(attrs)


def _build_span_limits(settings: SpanLimitSettings) -> SpanLimits | None:
    if not any(
        value is not None
        for value in (
            settings.attribute_count_limit,
            settings.attribute_value_length_limit,
            settings.span_event_count_limit,
            settings.span_link_count_limit,
            settings.event_attribute_count_limit,
            settings.link_attribute_count_limit,
        )
    ):
        return None
    return SpanLimits(
        max_attributes=settings.attribute_count_limit,
        max_attribute_length=settings.attribute_value_length_limit,
        max_events=settings.span_event_count_limit,
        max_links=settings.span_link_count_limit,
        max_event_attributes=settings.event_attribute_count_limit,
        max_link_attributes=settings.link_attribute_count_limit,
    )


def _build_sampler(config: ObservabilityConfig) -> Sampler | None:
    sampler = config.traces.sampler
    if not sampler:
        return None
    normalized = sampler.strip().lower()
    ratio = config.traces.sampler_arg if config.traces.sampler_arg is not None else 1.0
    mapping: dict[str, Sampler] = {
        "always_on": ALWAYS_ON,
        "always_off": ALWAYS_OFF,
        "parentbased_always_on": ParentBased(ALWAYS_ON),
        "parentbased_always_off": ParentBased(ALWAYS_OFF),
    }
    if normalized == "traceidratio":
        result: Sampler | None = TraceIdRatioBased(ratio)
    elif normalized == "parentbased_traceidratio":
        result = ParentBased(TraceIdRatioBased(ratio))
    else:
        result = mapping.get(normalized)
    if result is None:
        LOG.warning("Unsupported sampler %s; using SDK default", sampler)
    return result


def _build_exemplar_filter(config: ObservabilityConfig) -> ExemplarFilter | None:
    if not config.metrics.exemplar_filter:
        return None
    normalized = config.metrics.exemplar_filter.strip().lower()
    mapping: dict[str, ExemplarFilter] = {
        "always_on": AlwaysOnExemplarFilter(),
        "always_off": AlwaysOffExemplarFilter(),
        "trace_based": TraceBasedExemplarFilter(),
    }
    result = mapping.get(normalized)
    if result is None:
        LOG.warning(
            "Unsupported exemplar filter %s; using SDK default",
            config.metrics.exemplar_filter,
        )
    return result


def _build_views(config: ObservabilityConfig) -> list[View]:
    views: list[View] = []

    if config.metrics.views.operation_duration_ms_buckets:
        views.append(
            View(
                instrument_name="codeintel.operation.duration_ms",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metrics.views.operation_duration_ms_buckets)
                ),
            )
        )

    if config.metrics.views.query_duration_ms_buckets:
        views.append(
            View(
                instrument_name="codeintel.query.duration_ms",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metrics.views.query_duration_ms_buckets)
                ),
            )
        )

    if config.metrics.views.http_duration_s_buckets:
        views.append(
            View(
                instrument_name="http.server.request.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metrics.views.http_duration_s_buckets)
                ),
            )
        )

    if config.metrics.views.grpc_duration_s_buckets:
        views.append(
            View(
                instrument_name="grpc.client.call.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metrics.views.grpc_duration_s_buckets)
                ),
            )
        )
        views.append(
            View(
                instrument_name="grpc.server.call.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metrics.views.grpc_duration_s_buckets)
                ),
            )
        )

    return views


def build_metric_views(config: ObservabilityConfig) -> list[View]:
    """Build metric views for the supplied observability config.

    Returns
    -------
    list[View]
        Metric view definitions derived from the config.
    """
    return _build_views(config)


def build_exemplar_filter(config: ObservabilityConfig) -> ExemplarFilter | None:
    """Build an exemplar filter for the supplied observability config.

    Returns
    -------
    ExemplarFilter | None
        Exemplar filter derived from the config, if configured.
    """
    return _build_exemplar_filter(config)


def _load_module(name: str, *, label: str) -> object:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        message = f"{label} module unavailable: {exc}"
        raise RuntimeError(message) from exc


def _load_otel_logs_api() -> object:
    return _load_module("opentelemetry._logs", label="OpenTelemetry logs API")


def _load_sdk_logs_module() -> object:
    return _load_module("opentelemetry.sdk._logs", label="OpenTelemetry SDK logs")


def _load_sdk_logs_export_module() -> object:
    return _load_module("opentelemetry.sdk._logs.export", label="OpenTelemetry SDK logs export")


def _get_logger_provider_cls() -> type[_LoggerProvider]:
    module = _load_sdk_logs_module()
    provider_cls = getattr(module, "LoggerProvider", None)
    if not isinstance(provider_cls, type):
        message = "LoggerProvider class is unavailable in OpenTelemetry SDK logs"
        raise TypeError(message)
    return cast("type[_LoggerProvider]", provider_cls)


def _get_logging_handler_cls() -> type[logging.Handler]:
    module = _load_sdk_logs_module()
    handler_cls = getattr(module, "LoggingHandler", None)
    if not isinstance(handler_cls, type):
        message = "LoggingHandler class is unavailable in OpenTelemetry SDK logs"
        raise TypeError(message)
    return cast("type[logging.Handler]", handler_cls)


def _get_log_limits_cls() -> type[object]:
    module = _load_sdk_logs_module()
    limits_cls = getattr(module, "LogLimits", None)
    if not isinstance(limits_cls, type):
        message = "LogLimits class is unavailable in OpenTelemetry SDK logs"
        raise TypeError(message)
    return limits_cls


def _get_batch_log_record_processor_cls() -> type[object]:
    module = _load_sdk_logs_export_module()
    processor_cls = getattr(module, "BatchLogRecordProcessor", None)
    if not isinstance(processor_cls, type):
        message = "BatchLogRecordProcessor class is unavailable in OpenTelemetry SDK logs export"
        raise TypeError(message)
    return processor_cls


def _get_otlp_log_exporter_cls(protocol: str) -> type[object]:
    module_name = (
        "opentelemetry.exporter.otlp.proto.http._log_exporter"
        if protocol == "http/protobuf"
        else "opentelemetry.exporter.otlp.proto.grpc._log_exporter"
    )
    module = _load_module(module_name, label="OpenTelemetry OTLP log exporter")
    exporter_cls = getattr(module, "OTLPLogExporter", None)
    if not isinstance(exporter_cls, type):
        message = "OTLPLogExporter class is unavailable in OpenTelemetry exporters"
        raise TypeError(message)
    return exporter_cls


def _build_log_limits(settings: LogLimitSettings) -> object | None:
    if settings.attribute_count_limit is None and settings.attribute_value_length_limit is None:
        return None
    log_limits_ctor = cast("Callable[..., object]", _get_log_limits_cls())
    return log_limits_ctor(
        max_attributes=settings.attribute_count_limit,
        max_attribute_length=settings.attribute_value_length_limit,
    )


def _headers_to_dict(headers: tuple[tuple[str, str], ...]) -> dict[str, str] | None:
    if not headers:
        return None
    return dict(headers)


def _grpc_compression(value: str | None) -> grpc.Compression | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", ""}:
        return None
    if normalized == "gzip":
        return grpc.Compression.Gzip
    if normalized == "deflate":
        return grpc.Compression.Deflate
    LOG.warning("Unsupported gRPC compression %s; ignoring", value)
    return None


def _build_grpc_credentials(config: _ResolvedOtlp) -> grpc.ChannelCredentials | None:
    if not any([config.certificate, config.client_certificate, config.client_key]):
        return None
    try:
        root_cert = Path(config.certificate).read_bytes() if config.certificate else None
        client_cert = (
            Path(config.client_certificate).read_bytes() if config.client_certificate else None
        )
        client_key = Path(config.client_key).read_bytes() if config.client_key else None
    except OSError as exc:
        LOG.warning("Failed to load OTLP TLS credentials: %s", exc)
        return None
    return grpc.ssl_channel_credentials(
        root_certificates=root_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )


def _filter_kwargs(
    func: Callable[..., object],
    candidates: Mapping[str, object],
) -> dict[str, object]:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in candidates.items() if value is not None}
    return {key: value for key, value in candidates.items() if value is not None and key in params}


def _build_otlp_trace_exporter(config: _ResolvedOtlp) -> object:
    if config.protocol == "http/protobuf":
        exporter_cls = OTLPSpanExporterHttp
    else:
        exporter_cls = OTLPSpanExporterGrpc

    headers = _headers_to_dict(config.headers)
    kwargs: dict[str, object] = {
        "endpoint": config.endpoint,
        "headers": headers,
        "timeout": config.timeout_s,
    }

    if exporter_cls is OTLPSpanExporterGrpc:
        kwargs["compression"] = _grpc_compression(config.compression)
        kwargs["insecure"] = config.insecure
        kwargs["credentials"] = _build_grpc_credentials(config)
    else:
        kwargs["compression"] = config.compression
        kwargs["certificate_file"] = config.certificate
        kwargs["client_certificate_file"] = config.client_certificate
        kwargs["client_key_file"] = config.client_key

    filtered = _filter_kwargs(cast("Callable[..., object]", exporter_cls), kwargs)
    exporter_ctor = cast("Callable[..., object]", exporter_cls)
    return exporter_ctor(**filtered)


def _build_otlp_metric_exporter(config: _ResolvedOtlp) -> object:
    if config.protocol == "http/protobuf":
        exporter_cls = OTLPMetricExporterHttp
    else:
        exporter_cls = OTLPMetricExporterGrpc

    headers = _headers_to_dict(config.headers)
    kwargs: dict[str, object] = {
        "endpoint": config.endpoint,
        "headers": headers,
        "timeout": config.timeout_s,
    }

    if exporter_cls is OTLPMetricExporterGrpc:
        kwargs["compression"] = _grpc_compression(config.compression)
        kwargs["insecure"] = config.insecure
        kwargs["credentials"] = _build_grpc_credentials(config)
    else:
        kwargs["compression"] = config.compression
        kwargs["certificate_file"] = config.certificate
        kwargs["client_certificate_file"] = config.client_certificate
        kwargs["client_key_file"] = config.client_key

    filtered = _filter_kwargs(cast("Callable[..., object]", exporter_cls), kwargs)
    exporter_ctor = cast("Callable[..., object]", exporter_cls)
    return exporter_ctor(**filtered)


def _build_otlp_log_exporter(config: _ResolvedOtlp) -> object:
    exporter_cls = _get_otlp_log_exporter_cls(config.protocol)

    headers = _headers_to_dict(config.headers)
    kwargs: dict[str, object] = {
        "endpoint": config.endpoint,
        "headers": headers,
        "timeout": config.timeout_s,
    }

    if config.protocol != "http/protobuf":
        kwargs["compression"] = _grpc_compression(config.compression)
        kwargs["insecure"] = config.insecure
        kwargs["credentials"] = _build_grpc_credentials(config)
    else:
        kwargs["compression"] = config.compression
        kwargs["certificate_file"] = config.certificate
        kwargs["client_certificate_file"] = config.client_certificate
        kwargs["client_key_file"] = config.client_key

    filtered = _filter_kwargs(cast("Callable[..., object]", exporter_cls), kwargs)
    exporter_ctor = cast("Callable[..., object]", exporter_cls)
    return exporter_ctor(**filtered)


def _build_batch_kwargs(settings: BatchProcessorSettings) -> dict[str, object]:
    return {
        "schedule_delay_millis": settings.schedule_delay_ms,
        "max_queue_size": settings.max_queue_size,
        "max_export_batch_size": settings.max_export_batch_size,
        "export_timeout_millis": settings.export_timeout_ms,
    }


def _build_metric_reader_kwargs(settings: MetricExportSettings) -> dict[str, object]:
    return {
        "export_interval_millis": settings.export_interval_ms,
        "export_timeout_millis": settings.export_timeout_ms,
    }


def _build_tracer_provider(config: ObservabilityConfig, resource: Resource) -> TracerProvider:
    span_limits = _build_span_limits(config.traces.span_limits)
    sampler = _build_sampler(config)
    tracer_provider = TracerProvider(
        sampler=sampler,
        resource=resource,
        span_limits=span_limits,
    )

    if config.traces.enabled:
        resolved = _resolve_otlp(config.exporters.otlp, config.exporters.otlp_traces)
        exporter = _build_otlp_trace_exporter(resolved)
        processor_kwargs = _filter_kwargs(
            cast("Callable[..., object]", BatchSpanProcessor),
            _build_batch_kwargs(config.traces.batch),
        )
        processor_ctor = cast("Callable[..., object]", BatchSpanProcessor)
        processor = cast("SpanProcessor", processor_ctor(exporter, **processor_kwargs))
        tracer_provider.add_span_processor(processor)

    if config.traces.console_export:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if config.db_tracing.query_summary_span_name_hook:
        processor = _db_query_summary_span_name_processor()
        if processor is not None:
            tracer_provider.add_span_processor(processor)

    return tracer_provider


def _build_meter_provider(
    config: ObservabilityConfig,
    resource: Resource,
) -> tuple[MeterProvider, bool]:
    metric_readers: list[MetricReader] = []

    if config.metrics.enabled:
        resolved = _resolve_otlp(config.exporters.otlp, config.exporters.otlp_metrics)
        exporter = _build_otlp_metric_exporter(resolved)
        reader_kwargs = _filter_kwargs(
            cast("Callable[..., object]", PeriodicExportingMetricReader),
            _build_metric_reader_kwargs(config.metrics.export),
        )
        reader_ctor = cast("Callable[..., object]", PeriodicExportingMetricReader)
        reader = cast("MetricReader", reader_ctor(exporter, **reader_kwargs))
        metric_readers.append(reader)

    prometheus_enabled = False
    if config.metrics.prometheus_enabled:
        metric_readers.append(PrometheusMetricReader())
        prometheus_enabled = True

    views = _build_views(config)
    exemplar_filter = _build_exemplar_filter(config)
    meter_provider = MeterProvider(
        metric_readers=metric_readers,
        resource=resource,
        exemplar_filter=exemplar_filter,
        views=views,
    )
    return meter_provider, prometheus_enabled


def _build_logger_provider(
    config: ObservabilityConfig,
    resource: Resource,
) -> tuple[_LoggerProvider | None, logging.Handler | None]:
    return _build_logger_provider_with_handler(
        config,
        resource,
        force_handler=False,
    )


def _build_logger_provider_with_handler(
    config: ObservabilityConfig,
    resource: Resource,
    *,
    force_handler: bool,
) -> tuple[_LoggerProvider | None, logging.Handler | None]:
    if not config.logs.enabled and not config.logs.auto_instrument and not force_handler:
        return None, None

    log_limits = _build_log_limits(config.logs.limits)
    logger_kwargs: dict[str, object] = {"resource": resource}
    if log_limits is not None:
        logger_kwargs["log_record_limits"] = log_limits

    logger_provider_cls = _get_logger_provider_cls()
    logger_provider_ctor = cast("Callable[..., _LoggerProvider]", logger_provider_cls)
    filtered_logger_kwargs = _filter_kwargs(
        cast("Callable[..., object]", logger_provider_cls),
        logger_kwargs,
    )
    logger_provider = logger_provider_ctor(**filtered_logger_kwargs)

    if config.logs.enabled:
        resolved = _resolve_otlp(config.exporters.otlp, config.exporters.otlp_logs)
        exporter = _build_otlp_log_exporter(resolved)
        batch_processor_cls = _get_batch_log_record_processor_cls()
        processor_kwargs = _filter_kwargs(
            cast("Callable[..., object]", batch_processor_cls),
            _build_batch_kwargs(config.logs.batch),
        )
        batch_processor_ctor = cast("Callable[..., object]", batch_processor_cls)
        logger_provider.add_log_record_processor(batch_processor_ctor(exporter, **processor_kwargs))

    handler_cls = _get_logging_handler_cls()
    handler_ctor = cast("Callable[..., logging.Handler]", handler_cls)
    log_handler = handler_ctor(level=logging.NOTSET, logger_provider=logger_provider)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    if config.logs.trace_filter:
        log_handler.addFilter(_trace_sampled_log_filter())

    return logger_provider, log_handler


def _instrument_logging(config: ObservabilityConfig, registry: InstrumentationRegistry) -> None:
    if config.logs.correlation or config.logs.auto_instrument:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO)
        elif root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    def _instrument(name: str, instrumentor: object, **kwargs: object) -> None:
        instrument = getattr(instrumentor, "instrument", None)
        if not callable(instrument):
            registry.record_unavailable(name, detail="Instrumentor missing instrument()")
            return
        try:
            instrument(**kwargs)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            registry.record_error(name, str(exc))
        else:
            registry.record_enabled(name)

    if config.logs.correlation or config.logs.auto_instrument:
        logging_instrumentor = LoggingInstrumentor()
        instrument = getattr(logging_instrumentor, "instrument", None)
        if callable(instrument):
            logging_kwargs = _filter_kwargs(
                instrument,
                {"set_logging_format": config.logs.auto_instrument},
            )
        else:
            logging_kwargs = {}
        _instrument("logging", logging_instrumentor, **logging_kwargs)
    else:
        registry.record_suppressed("logging")


def _instrument_runtime(registry: InstrumentationRegistry) -> None:
    def _instrument(name: str, instrumentor: object, **kwargs: object) -> None:
        instrument = getattr(instrumentor, "instrument", None)
        if not callable(instrument):
            registry.record_unavailable(name, detail="Instrumentor missing instrument()")
            return
        try:
            instrument(**kwargs)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            registry.record_error(name, str(exc))
        else:
            registry.record_enabled(name)

    _instrument("threading", ThreadingInstrumentor())
    _instrument("asyncio", AsyncioInstrumentor())
    _instrument("httpx", HTTPXClientInstrumentor())
    _instrument("requests", RequestsInstrumentor())


def _configure_propagators(config: ObservabilityConfig) -> None:
    if not config.resources.propagators:
        return

    propagators: list[TextMapPropagator] = []
    for name in config.resources.propagators:
        normalized = name.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            propagators.clear()
            break
        if normalized == "tracecontext":
            propagators.append(TraceContextTextMapPropagator())
            continue
        if normalized == "baggage":
            propagators.append(W3CBaggagePropagator())
            continue
        propagator = _load_optional_propagator(normalized)
        if propagator is None:
            LOG.warning("Unsupported propagator %s; skipping", name)
            continue
        propagators.append(propagator)

    if propagators:
        set_global_textmap(CompositePropagator(propagators))


def _load_optional_propagator(name: str) -> TextMapPropagator | None:
    if name in {"b3", "b3multi"}:
        return _load_propagator("opentelemetry.propagators.b3", "B3MultiFormat")
    if name == "b3single":
        return _load_propagator("opentelemetry.propagators.b3", "B3SingleFormat")
    if name == "jaeger":
        return _load_propagator("opentelemetry.propagators.jaeger", "JaegerPropagator")
    if name == "ottrace":
        return _load_propagator("opentelemetry.propagators.ot_trace", "OTTracePropagator")
    return None


def _load_propagator(module_name: str, symbol: str) -> TextMapPropagator | None:
    try:
        module = __import__(module_name, fromlist=[symbol])
    except ImportError:
        return None
    value = getattr(module, symbol, None)
    if value is None:
        return None
    instance = value() if callable(value) else None
    if isinstance(instance, TextMapPropagator):
        return instance
    return None


def _trace_sampled_log_filter() -> logging.Filter:
    class _Filter(logging.Filter):
        @staticmethod
        def filter(record: logging.LogRecord) -> bool:
            _ = record
            span = otel_trace.get_current_span()
            if span is None:
                return True
            context = span.get_span_context()
            if context is None:
                return True
            return bool(context.trace_flags.sampled)

    return _Filter()


def _shutdown_component(
    component: object | None,
    *,
    label: str,
    errors: list[str],
    timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
) -> bool:
    if component is None:
        return True
    shutdown = getattr(component, "shutdown", None)
    if not callable(shutdown):
        return True
    shutdown_func = cast("Callable[[], object]", shutdown)
    done = threading.Event()
    thread_errors: list[str] = []

    def _run() -> None:
        try:
            shutdown_func()
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            thread_errors.append(f"{label}:{exc}")
        finally:
            done.set()

    thread = threading.Thread(
        target=_run,
        name=f"otel-shutdown-{label}",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout_s):
        errors.append(f"{label}:shutdown-timeout")
        return False
    if thread_errors:
        errors.extend(thread_errors)
        return False
    return True


def _disabled_runtime() -> ObservabilityRuntime:
    config = ObservabilityConfig(enabled=False)
    return ObservabilityRuntime(
        enabled=False,
        tracer=None,
        meter=None,
        logger_provider=None,
        log_handler=None,
        shutdown=None,
        policy=ObservabilityPolicy(),
        prometheus_enabled=False,
        grpc_observability=None,
        db_tracing=DbTracingConfig(enabled=False),
        config=config,
        config_provenance=ConfigProvenance.unknown_for(config),
    )


def _build_shutdown(
    tracer_provider: TracerProvider | None,
    meter_provider: MeterProvider | None,
    logger_provider: _LoggerProvider | None,
    *,
    log_handler: logging.Handler | None,
    grpc_handle: GrpcObservabilityHandle | None,
) -> Callable[[], ObservabilityShutdownResult]:
    def _shutdown() -> ObservabilityShutdownResult:
        start = time.perf_counter()
        errors: list[str] = []
        flush_ok = _shutdown_component(grpc_handle, label="grpc", errors=errors)

        if log_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(log_handler)

        flush_ok = _shutdown_component(tracer_provider, label="tracer", errors=errors) and flush_ok
        flush_ok = _shutdown_component(meter_provider, label="meter", errors=errors) and flush_ok
        flush_ok = _shutdown_component(logger_provider, label="logger", errors=errors) and flush_ok

        duration_ms = (time.perf_counter() - start) * 1000
        return ObservabilityShutdownResult(
            flush_ok=flush_ok,
            flush_ms=duration_ms,
            errors=tuple(errors),
        )

    return _shutdown


@dataclass(frozen=True, slots=True)
class RuntimeGlobalHandles:
    """Dependencies required to build a runtime from global providers."""

    log_handler: logging.Handler | None
    logger_provider: _LoggerProvider | None
    grpc_handle: GrpcObservabilityHandle | None
    prometheus_enabled: bool
    provenance: ConfigProvenance | None = None


def _runtime_from_global(
    config: ObservabilityConfig,
    *,
    handles: RuntimeGlobalHandles,
) -> ObservabilityRuntime:
    tracer_provider = cast("TracerProvider", otel_trace.get_tracer_provider())
    meter_provider = cast("MeterProvider", otel_metrics.get_meter_provider())
    tracer = otel_trace.get_tracer(config.resources.service_name)
    meter = otel_metrics.get_meter(config.resources.service_name)
    shutdown = _build_shutdown(
        tracer_provider,
        meter_provider,
        handles.logger_provider,
        log_handler=handles.log_handler,
        grpc_handle=handles.grpc_handle,
    )
    return ObservabilityRuntime(
        enabled=True,
        tracer=tracer,
        meter=meter,
        logger_provider=handles.logger_provider,
        log_handler=handles.log_handler,
        shutdown=shutdown,
        policy=config.policy,
        prometheus_enabled=handles.prometheus_enabled,
        grpc_observability=handles.grpc_handle,
        db_tracing=config.db_tracing,
        config=config,
        config_provenance=handles.provenance,
    )


def _apply_collector_local_config(config: ObservabilityConfig) -> ObservabilityConfig:
    local_otlp = replace(config.exporters.otlp, endpoint="http://localhost:4317", protocol="grpc")
    local_traces = replace(
        config.exporters.otlp_traces,
        endpoint="http://localhost:4317",
        protocol="grpc",
    )
    local_metrics = replace(
        config.exporters.otlp_metrics,
        endpoint="http://localhost:4317",
        protocol="grpc",
    )
    local_logs = replace(
        config.exporters.otlp_logs,
        endpoint="http://localhost:4317",
        protocol="grpc",
    )
    exporters = replace(
        config.exporters,
        otlp=local_otlp,
        otlp_traces=local_traces,
        otlp_metrics=local_metrics,
        otlp_logs=local_logs,
    )
    return replace(
        config,
        exporters=exporters,
        config_file=None,
        traces=replace(config.traces, console_export=False),
        metrics=replace(config.metrics, prometheus_enabled=False),
    )


def _is_sdk_tracer_provider(provider: object) -> bool:
    return hasattr(provider, "add_span_processor") and hasattr(provider, "shutdown")


def _is_sdk_meter_provider(provider: object) -> bool:
    return hasattr(provider, "force_flush") and hasattr(provider, "shutdown")


def _runtime_from_in_memory(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
    *,
    provenance: ConfigProvenance | None = None,
) -> ObservabilityRuntime:
    resource = _build_resource(config)
    span_limits = _build_span_limits(config.traces.span_limits)
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(
        sampler=ALWAYS_ON,
        resource=resource,
        span_limits=span_limits,
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(config.resources.service_name)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
        resource=resource,
        exemplar_filter=_build_exemplar_filter(config),
        views=_build_views(config),
    )
    otel_metrics.set_meter_provider(meter_provider)
    meter = otel_metrics.get_meter(config.resources.service_name)

    logging_pipeline = LoggingPipeline(config=config, resource=resource, registry=registry)
    logging_state = logging_pipeline.install(force_handler=False)
    logger_provider = logging_state.logger_provider
    log_handler = logging_state.log_handler
    if logger_provider is not None:
        logs_api = _load_otel_logs_api()
        set_logger_provider = getattr(logs_api, "set_logger_provider", None)
        if callable(set_logger_provider):
            set_logger_provider(logger_provider)

    _instrument_runtime(registry)

    registry.emit_summary(policy=config.policy, logger=LOG)
    registry.emit_metrics(meter, policy=config.policy)

    shutdown = _build_shutdown(
        tracer_provider,
        meter_provider,
        logger_provider=logger_provider,
        log_handler=log_handler,
        grpc_handle=None,
    )

    return ObservabilityRuntime(
        enabled=True,
        tracer=tracer,
        meter=meter,
        logger_provider=logger_provider,
        log_handler=log_handler,
        shutdown=shutdown,
        policy=config.policy,
        prometheus_enabled=False,
        grpc_observability=None,
        db_tracing=config.db_tracing,
        test_handles=TestTelemetryHandles(
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        ),
        config=config,
        config_provenance=provenance,
    )


def _runtime_from_config_file(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
    *,
    provenance: ConfigProvenance | None = None,
) -> ObservabilityRuntime | None:
    if not config.config_file:
        return None

    try:
        load_otel_config_file(config.config_file)
    except (ValueError, FileNotFoundError) as exc:
        message = f"Invalid OpenTelemetry config file: {exc}"
        raise RuntimeError(message) from exc

    try:
        configured = apply_otel_config_file(config.config_file)
    except RuntimeError as exc:
        LOG.warning("Failed to apply OTEL config file %s: %s", config.config_file, exc)
        return None
    if not configured:
        LOG.warning(
            "Failed to apply OTEL config file %s; using manual config",
            config.config_file,
        )
        return None

    tracer_provider = otel_trace.get_tracer_provider()
    meter_provider = otel_metrics.get_meter_provider()
    if not _is_sdk_tracer_provider(tracer_provider) or not _is_sdk_meter_provider(meter_provider):
        LOG.warning(
            "OTEL config file did not initialize SDK providers; falling back to manual config"
        )
        return None

    _instrument_logging(config, registry)
    _instrument_runtime(registry)
    logger_provider: _LoggerProvider | None = None
    log_handler: logging.Handler | None = None
    try:
        logs_api = _load_otel_logs_api()
    except RuntimeError:
        logs_api = None
    if logs_api is not None:
        get_logger_provider = getattr(logs_api, "get_logger_provider", None)
        if callable(get_logger_provider):
            logger_provider = cast("_LoggerProvider", get_logger_provider())
            handler_cls = _get_logging_handler_cls()
            handler_ctor = cast("Callable[..., logging.Handler]", handler_cls)
            log_handler = handler_ctor(
                level=logging.NOTSET,
                logger_provider=logger_provider,
            )
            logging.getLogger().addHandler(log_handler)
            if config.logs.trace_filter:
                log_handler.addFilter(_trace_sampled_log_filter())

    grpc_handle = register_grpc_observability(
        config.grpc_observability,
        meter_provider=otel_metrics.get_meter_provider(),
        registry=registry,
    )
    registry.emit_summary(policy=config.policy, logger=LOG)
    registry.emit_metrics(
        otel_metrics.get_meter(config.resources.service_name),
        policy=config.policy,
    )
    return _runtime_from_global(
        config,
        handles=RuntimeGlobalHandles(
            log_handler=log_handler,
            logger_provider=logger_provider,
            grpc_handle=grpc_handle,
            prometheus_enabled=config.metrics.prometheus_enabled,
            provenance=provenance,
        ),
    )


def _runtime_from_manual_config(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
    *,
    force_log_handler: bool = False,
    provenance: ConfigProvenance | None = None,
) -> ObservabilityRuntime:
    resource = _build_resource(config)

    tracer_provider = _build_tracer_provider(config, resource)
    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(config.resources.service_name)

    meter_provider, prometheus_enabled = _build_meter_provider(config, resource)
    otel_metrics.set_meter_provider(meter_provider)
    meter = otel_metrics.get_meter(config.resources.service_name)

    logging_pipeline = LoggingPipeline(config=config, resource=resource, registry=registry)
    logging_state = logging_pipeline.install(force_handler=force_log_handler)
    logger_provider = logging_state.logger_provider
    log_handler = logging_state.log_handler
    if logger_provider is not None:
        logs_api = _load_otel_logs_api()
        set_logger_provider = getattr(logs_api, "set_logger_provider", None)
        if callable(set_logger_provider):
            set_logger_provider(logger_provider)

    _instrument_runtime(registry)

    grpc_handle = register_grpc_observability(
        config.grpc_observability,
        meter_provider=meter_provider,
        registry=registry,
    )

    registry.emit_summary(policy=config.policy, logger=LOG)
    registry.emit_metrics(meter, policy=config.policy)

    shutdown = _build_shutdown(
        tracer_provider,
        meter_provider,
        logger_provider,
        log_handler=log_handler,
        grpc_handle=grpc_handle,
    )

    return ObservabilityRuntime(
        enabled=True,
        tracer=tracer,
        meter=meter,
        logger_provider=logger_provider,
        log_handler=log_handler,
        shutdown=shutdown,
        policy=config.policy,
        prometheus_enabled=prometheus_enabled,
        grpc_observability=grpc_handle,
        db_tracing=config.db_tracing,
        config=config,
        config_provenance=provenance,
    )


def _init_observability(resolved: ResolvedObservabilityConfig) -> ObservabilityRuntime:
    config = resolved.config
    provenance = resolved.provenance
    test_mode = config.test_mode
    if test_mode == "disabled":
        return _disabled_runtime()

    if config.config_file is None:
        _configure_propagators(config)

    registry = get_instrumentation_registry()
    if test_mode == "in_memory":
        return _runtime_from_in_memory(config, registry, provenance=provenance)

    if test_mode == "collector_local":
        config = _apply_collector_local_config(config)

    if not config.enabled:
        return _disabled_runtime()

    runtime = _runtime_from_config_file(config, registry, provenance=provenance)
    if runtime is not None:
        return runtime
    return _runtime_from_manual_config(
        config,
        registry,
        force_log_handler=config.config_file is not None,
        provenance=provenance,
    )


def bootstrap_observability(
    config: ObservabilityConfig | ResolvedObservabilityConfig,
) -> ObservabilityRuntime:
    """Initialize OpenTelemetry providers (idempotent).

    Returns
    -------
    ObservabilityRuntime
        Initialized observability runtime handles.
    """
    return _RUNTIME_MANAGER.bootstrap(config)


def get_observability() -> ObservabilityRuntime:
    """Return the active observability runtime state.

    Returns
    -------
    ObservabilityRuntime
        Active runtime handles or a disabled runtime when uninitialized.
    """
    return _RUNTIME_MANAGER.get()


def shutdown_observability() -> ObservabilityShutdownResult | None:
    """Shut down the active observability runtime, if available.

    Returns
    -------
    ObservabilityShutdownResult | None
        Structured flush results, or None if observability is inactive.
    """
    return _RUNTIME_MANAGER.shutdown()


def flush_observability() -> ObservabilityShutdownResult | None:
    """Force-flush the active observability runtime without shutting it down.

    Returns
    -------
    ObservabilityShutdownResult | None
        Structured flush results, or None if observability is inactive.
    """
    return _RUNTIME_MANAGER.flush()


def get_runtime_manager() -> ObservabilityRuntimeManager:
    """Return the global observability runtime manager.

    Returns
    -------
    ObservabilityRuntimeManager
        Global runtime manager instance.
    """
    return _RUNTIME_MANAGER


def _log_shutdown_result(
    result: ObservabilityShutdownResult,
    *,
    policy: ObservabilityPolicy,
) -> None:
    payload = {
        TELEMETRY_FLUSH_OK: result.flush_ok,
        TELEMETRY_FLUSH_MS: result.flush_ms,
        "errors": list(result.errors),
    }
    event = TelemetryEvent(
        name="telemetry.flush",
        span_attributes={
            TELEMETRY_FLUSH_OK: result.flush_ok,
            TELEMETRY_FLUSH_MS: result.flush_ms,
        },
        log_payload=payload,
        log_level=logging.INFO if result.flush_ok else logging.WARNING,
    )
    emit_event(
        event=event,
        span=None,
        normalizer=build_attribute_normalizer(policy),
        logger=LOG,
    )


@dataclass(slots=True)
class _PipelineInstruments:
    flush_attempts: Counter
    flush_failures: Counter
    flush_duration_ms: Histogram


_PIPELINE_INSTRUMENTS: WeakKeyDictionary[Meter, _PipelineInstruments] = WeakKeyDictionary()


def _get_pipeline_instruments(meter: Meter) -> _PipelineInstruments:
    instruments = _PIPELINE_INSTRUMENTS.get(meter)
    if instruments is not None:
        return instruments
    instruments = _PipelineInstruments(
        flush_attempts=meter.create_counter(
            "codeintel.telemetry.flush.attempts",
            unit="1",
            description="Count of telemetry flush attempts",
        ),
        flush_failures=meter.create_counter(
            "codeintel.telemetry.flush.failures",
            unit="1",
            description="Count of telemetry flush failures",
        ),
        flush_duration_ms=meter.create_histogram(
            "codeintel.telemetry.flush.duration_ms",
            unit="ms",
            description="Duration of telemetry flush operations (ms)",
        ),
    )
    _PIPELINE_INSTRUMENTS[meter] = instruments
    return instruments


def _record_pipeline_metrics(
    result: ObservabilityShutdownResult,
    *,
    action: str,
) -> None:
    runtime = _ObservabilityHolder.get_or_none()
    if runtime is None or runtime.meter is None:
        return
    instruments = _get_pipeline_instruments(runtime.meter)
    normalizer = build_attribute_normalizer(runtime.policy)
    attrs = normalizer.normalize(
        {TELEMETRY_ACTION: action},
        allowed_keys=frozenset({TELEMETRY_ACTION}),
    )
    instruments.flush_attempts.add(1, attributes=attrs)
    if not result.flush_ok:
        instruments.flush_failures.add(1, attributes=attrs)
    instruments.flush_duration_ms.record(result.flush_ms, attributes=attrs)


def _record_pipeline_health(result: ObservabilityShutdownResult) -> None:
    _PIPELINE_HEALTH_STATE.last_flush_ok = result.flush_ok
    _PIPELINE_HEALTH_STATE.last_flush_ms = result.flush_ms
    _PIPELINE_HEALTH_STATE.last_flush_errors = result.errors


def get_pipeline_health_state() -> PipelineHealthState:
    """Return the last telemetry pipeline flush summary.

    Returns
    -------
    PipelineHealthState
        Snapshot of the last pipeline flush attempt.
    """
    return PipelineHealthState(
        last_flush_ok=_PIPELINE_HEALTH_STATE.last_flush_ok,
        last_flush_ms=_PIPELINE_HEALTH_STATE.last_flush_ms,
        last_flush_errors=_PIPELINE_HEALTH_STATE.last_flush_errors,
    )


def _observability_config_from_settings(
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
    resources = ResourceConfig(
        service_name=service_name,
        service_version=settings.service_version,
        deployment_environment=settings.deployment_environment,
        resource_attributes=settings.resource_attributes,
        propagators=settings.propagators,
    )
    exporters = ExporterConfig(
        otlp=settings.otlp,
        otlp_traces=settings.otlp_traces,
        otlp_metrics=settings.otlp_metrics,
        otlp_logs=settings.otlp_logs,
    )
    traces = TraceConfig(
        enabled=settings.export_traces,
        sampler=settings.traces_sampler,
        sampler_arg=settings.traces_sampler_arg,
        batch=settings.traces_batch,
        span_limits=settings.span_limits,
        console_export=settings.console_export,
    )
    metrics = MetricConfig(
        enabled=settings.export_metrics,
        export=settings.metrics_export,
        exemplar_filter=settings.metrics_exemplar_filter,
        views=settings.metric_views,
        prometheus_enabled=settings.prometheus_enabled,
    )
    logs = LogConfig(
        enabled=settings.export_logs,
        auto_instrument=settings.logs_auto_instrument,
        correlation=settings.log_correlation,
        trace_filter=settings.logs_trace_filter,
        batch=settings.logs_batch,
        limits=settings.log_limits,
    )
    cli = CliTelemetryConfig(
        enabled=settings.cli_enabled,
        args_allowlist=settings.cli_args_allowlist,
        args_capture_mode=settings.cli_args_capture_mode,
    )
    teardown = TeardownConfig(
        enabled=settings.teardown_enabled,
        task_sample_limit=settings.teardown_task_sample_limit,
        thread_sample_limit=settings.teardown_thread_sample_limit,
        subprocess_sample_limit=settings.teardown_subprocess_sample_limit,
    )
    db_tracing = DbTracingConfig(
        enabled=settings.duckdb_tracing_enabled,
        require_parent_span=settings.duckdb_require_parent_span,
        statement_mode=settings.duckdb_statement_mode,
        statement_hash_len=settings.duckdb_statement_hash_len,
        query_summary_max_len=settings.duckdb_query_summary_max_len,
        query_summary_max_targets=settings.duckdb_query_summary_max_targets,
        query_summary_emit_ellipsis=settings.duckdb_query_summary_emit_ellipsis,
        query_summary_hash_suspicious_targets=(
            settings.duckdb_query_summary_hash_suspicious_targets
        ),
        query_summary_hash_len=settings.duckdb_query_summary_hash_len,
        query_summary_hash_min_len=settings.duckdb_query_summary_hash_min_len,
        query_summary_include_subquery_operations=(
            settings.duckdb_query_summary_include_subquery_operations
        ),
        query_summary_include_multi_statement=settings.duckdb_query_summary_include_multi_statement,
        query_summary_span_name_hook=settings.db_query_summary_span_name_hook,
        query_text_policy=settings.duckdb_query_text_policy,
        query_text_max_len=settings.duckdb_query_text_max_len,
        query_text_strip_comments=settings.duckdb_query_text_strip_comments,
        query_text_collapse_in_lists=settings.duckdb_query_text_collapse_in_lists,
        query_parameter_enabled=settings.duckdb_query_parameter_enabled,
        query_parameter_keys=settings.duckdb_query_parameter_keys,
        query_parameter_hash_keys=settings.duckdb_query_parameter_hash_keys,
        query_parameter_require_in_sql=settings.duckdb_query_parameter_require_in_sql,
        query_parameter_max_str_len=settings.duckdb_query_parameter_max_str_len,
    )
    return ObservabilityConfig(
        enabled=settings.enabled,
        resources=resources,
        exporters=exporters,
        traces=traces,
        metrics=metrics,
        logs=logs,
        cli=cli,
        teardown=teardown,
        grpc_observability=settings.grpc_observability,
        hamilton_tracker=settings.hamilton_tracker,
        db_tracing=db_tracing,
        config_file=settings.config_file,
        test_mode=resolve_test_telemetry_mode(),
        policy=policy_from_settings(settings),
    )


def observability_config_from_settings(
    settings: ObservabilitySettings,
    *,
    default_service_name: str,
    overrides: Mapping[str, object] | None = None,
) -> ObservabilityConfig:
    """Build observability configuration from settings.

    Returns
    -------
    ObservabilityConfig
        Resolved observability configuration.
    """
    resolved = resolve_observability_config(
        settings,
        default_service_name=default_service_name,
        overrides=overrides,
    )
    return resolved.config


def resolve_observability_config(
    settings: ObservabilitySettings,
    *,
    default_service_name: str,
    overrides: Mapping[str, object] | None = None,
) -> ResolvedObservabilityConfig:
    """Resolve an observability config and provenance snapshot.

    Returns
    -------
    ResolvedObservabilityConfig
        Resolved config with provenance metadata.
    """
    resolver = ConfigResolver(default_service_name=default_service_name)
    return resolver.resolve(settings, overrides=overrides)


def _coerce_resolved_config(
    config: ObservabilityConfig | ResolvedObservabilityConfig,
) -> ResolvedObservabilityConfig:
    if isinstance(config, ResolvedObservabilityConfig):
        return config
    return ResolvedObservabilityConfig(
        config=config,
        provenance=ConfigProvenance.unknown_for(config),
    )


def _apply_overrides(
    config: ObservabilityConfig,
    overrides: Mapping[str, object],
) -> ObservabilityConfig:
    updated = config
    for path, value in overrides.items():
        updated = cast("ObservabilityConfig", _replace_path(updated, path.split("."), value))
    return updated


def _replace_path(obj: object, parts: list[str], value: object) -> object:
    if not parts:
        return obj
    if not is_dataclass(obj):
        message = f"Cannot apply override to non-dataclass at {'.'.join(parts)}"
        raise TypeError(message)
    dataclass_obj = cast("_DataclassFields", obj)
    field_name = parts[0]
    if field_name not in dataclass_obj.__dataclass_fields__:
        message = f"Unknown config field: {field_name}"
        raise AttributeError(message)
    if len(parts) == 1:
        return _replace_dataclass_value(dataclass_obj, field_name, value)
    nested = getattr(dataclass_obj, field_name)
    replaced = _replace_path(nested, parts[1:], value)
    return _replace_dataclass_value(dataclass_obj, field_name, replaced)


def _replace_dataclass_value(
    obj: _DataclassFields,
    field_name: str,
    value: object,
) -> object:
    values = {name: getattr(obj, name) for name in obj.__dataclass_fields__}
    values[field_name] = value
    constructor = cast("Callable[..., object]", obj.__class__)
    return constructor(**values)


def _flatten_config(config: ObservabilityConfig) -> dict[str, object]:
    flattened: dict[str, object] = {}

    def _walk(value: object, prefix: str) -> None:
        if is_dataclass(value):
            for field_def in value.__dataclass_fields__.values():
                field_value = getattr(value, field_def.name)
                key = f"{prefix}.{field_def.name}" if prefix else field_def.name
                _walk(field_value, key)
        else:
            flattened[prefix] = value

    _walk(config, "")
    return flattened


def _serialize_config(config: ObservabilityConfig) -> dict[str, object]:
    def _serialize(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {
                field_name: _serialize(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
        if isinstance(value, Mapping):
            return {str(key): _serialize(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return [_serialize(item) for item in value]
        if isinstance(value, (set, frozenset)):
            serialized = [_serialize(item) for item in value]
            return sorted(serialized, key=str)
        return value

    return cast("dict[str, object]", _serialize(config))


__all__ = [
    "CliTelemetryConfig",
    "ConfigProvenance",
    "ConfigResolver",
    "ConfigSource",
    "DbTracingConfig",
    "ExporterConfig",
    "LogConfig",
    "LoggingPipeline",
    "LoggingPipelineState",
    "MetricConfig",
    "ObservabilityConfig",
    "ObservabilityRuntime",
    "ObservabilityRuntimeManager",
    "ObservabilityShutdownResult",
    "PipelineHealthState",
    "ResolvedObservabilityConfig",
    "ResourceConfig",
    "TeardownConfig",
    "TestTelemetryHandles",
    "TraceConfig",
    "bootstrap_observability",
    "build_exemplar_filter",
    "build_metric_views",
    "flush_observability",
    "get_observability",
    "get_pipeline_health_state",
    "get_runtime_manager",
    "observability_config_from_settings",
    "resolve_observability_config",
    "shutdown_observability",
]
