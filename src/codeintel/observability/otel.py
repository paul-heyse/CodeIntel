"""OpenTelemetry bootstrap and shared runtime access."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

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
from codeintel.observability.attributes import shape_attributes
from codeintel.observability.config_validation import validate_otel_config_file
from codeintel.observability.policy import ObservabilityPolicy, policy_from_settings
from codeintel.observability.grpc import GrpcObservabilityHandle, register_grpc_observability
from codeintel.observability.instrumentation_registry import (
    InstrumentationRegistry,
    get_instrumentation_registry,
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


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Runtime configuration for OpenTelemetry bootstrap."""

    enabled: bool = True
    service_name: str = "codeintel"
    service_version: str | None = None
    deployment_environment: str | None = None
    resource_attributes: tuple[tuple[str, str], ...] = ()
    propagators: tuple[str, ...] = ()
    traces_sampler: str | None = None
    traces_sampler_arg: float | None = None
    config_file: Path | None = None
    otlp: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_traces: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_metrics: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_logs: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    export_traces: bool = True
    export_metrics: bool = True
    export_logs: bool = False
    console_export: bool = False
    prometheus_enabled: bool = False
    logs_auto_instrument: bool = False
    log_correlation: bool = False
    logs_trace_filter: bool = False
    traces_batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    logs_batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    metrics_export: MetricExportSettings = field(default_factory=MetricExportSettings)
    span_limits: SpanLimitSettings = field(default_factory=SpanLimitSettings)
    log_limits: LogLimitSettings = field(default_factory=LogLimitSettings)
    metrics_exemplar_filter: str | None = None
    metric_views: MetricViewSettings = field(default_factory=MetricViewSettings)
    grpc_observability: GrpcObservabilitySettings = field(default_factory=GrpcObservabilitySettings)
    hamilton_tracker: HamiltonTrackerSettings = field(default_factory=HamiltonTrackerSettings)
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
    test_mode: TestTelemetryMode | None = None
    policy: ObservabilityPolicy = field(default_factory=ObservabilityPolicy)


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
    test_handles: TestTelemetryHandles | None = None


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

    def bootstrap(self, config: ObservabilityConfig) -> ObservabilityRuntime:
        """Initialize and return the observability runtime."""
        return _ObservabilityHolder.get(lambda: _init_observability(config))

    def get(self) -> ObservabilityRuntime:
        """Return the current runtime, or a disabled runtime."""
        runtime = _ObservabilityHolder.get_or_none()
        if runtime is not None:
            return runtime
        return _disabled_runtime()

    def shutdown(self) -> ObservabilityShutdownResult | None:
        """Shut down the runtime and reset state."""
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
            _record_pipeline_health(result)
            _record_pipeline_metrics(result, action="shutdown")
        _ObservabilityHolder.reset()
        return result

    def flush(self) -> ObservabilityShutdownResult | None:
        """Force-flush the runtime without shutdown."""
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
            flush_ok = _force_flush_provider(meter_provider, label="meter", errors=errors) and flush_ok
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

    def reset(self) -> None:
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
            "telemetry.flush.ok": self.flush_ok,
            "telemetry.flush.duration_ms": self.flush_ms,
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
    _SUMMARY_KEY = "db.query.summary"
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
    attrs = dict(config.resource_attributes)
    attrs.setdefault("service.name", config.service_name)
    service_version = config.service_version or _package_version()
    if service_version:
        attrs.setdefault("service.version", service_version)
    if config.deployment_environment:
        attrs.setdefault("deployment.environment.name", config.deployment_environment)
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
    sampler = config.traces_sampler
    if not sampler:
        return None
    normalized = sampler.strip().lower()
    ratio = config.traces_sampler_arg if config.traces_sampler_arg is not None else 1.0
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
    if not config.metrics_exemplar_filter:
        return None
    normalized = config.metrics_exemplar_filter.strip().lower()
    mapping: dict[str, ExemplarFilter] = {
        "always_on": AlwaysOnExemplarFilter(),
        "always_off": AlwaysOffExemplarFilter(),
        "trace_based": TraceBasedExemplarFilter(),
    }
    result = mapping.get(normalized)
    if result is None:
        LOG.warning(
            "Unsupported exemplar filter %s; using SDK default",
            config.metrics_exemplar_filter,
        )
    return result


def _build_views(config: ObservabilityConfig) -> list[View]:
    views: list[View] = []

    if config.metric_views.operation_duration_ms_buckets:
        views.append(
            View(
                instrument_name="codeintel.operation.duration_ms",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metric_views.operation_duration_ms_buckets)
                ),
            )
        )

    if config.metric_views.query_duration_ms_buckets:
        views.append(
            View(
                instrument_name="codeintel.query.duration_ms",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metric_views.query_duration_ms_buckets)
                ),
            )
        )

    if config.metric_views.http_duration_s_buckets:
        views.append(
            View(
                instrument_name="http.server.request.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metric_views.http_duration_s_buckets)
                ),
            )
        )

    if config.metric_views.grpc_duration_s_buckets:
        views.append(
            View(
                instrument_name="grpc.client.call.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metric_views.grpc_duration_s_buckets)
                ),
            )
        )
        views.append(
            View(
                instrument_name="grpc.server.call.duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    list(config.metric_views.grpc_duration_s_buckets)
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
    return {
        key: value
        for key, value in candidates.items()
        if value is not None and key in params
    }


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
    span_limits = _build_span_limits(config.span_limits)
    sampler = _build_sampler(config)
    tracer_provider = TracerProvider(
        sampler=sampler,
        resource=resource,
        span_limits=span_limits,
    )

    if config.export_traces:
        resolved = _resolve_otlp(config.otlp, config.otlp_traces)
        exporter = _build_otlp_trace_exporter(resolved)
        processor_kwargs = _filter_kwargs(
            cast("Callable[..., object]", BatchSpanProcessor),
            _build_batch_kwargs(config.traces_batch),
        )
        processor_ctor = cast("Callable[..., object]", BatchSpanProcessor)
        processor = cast("SpanProcessor", processor_ctor(exporter, **processor_kwargs))
        tracer_provider.add_span_processor(processor)

    if config.console_export:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if config.db_query_summary_span_name_hook:
        processor = _db_query_summary_span_name_processor()
        if processor is not None:
            tracer_provider.add_span_processor(processor)

    return tracer_provider


def _build_meter_provider(
    config: ObservabilityConfig,
    resource: Resource,
) -> tuple[MeterProvider, bool]:
    metric_readers: list[MetricReader] = []

    if config.export_metrics:
        resolved = _resolve_otlp(config.otlp, config.otlp_metrics)
        exporter = _build_otlp_metric_exporter(resolved)
        reader_kwargs = _filter_kwargs(
            cast("Callable[..., object]", PeriodicExportingMetricReader),
            _build_metric_reader_kwargs(config.metrics_export),
        )
        reader_ctor = cast("Callable[..., object]", PeriodicExportingMetricReader)
        reader = cast("MetricReader", reader_ctor(exporter, **reader_kwargs))
        metric_readers.append(reader)

    prometheus_enabled = False
    if config.prometheus_enabled:
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
    if not config.export_logs and not config.logs_auto_instrument:
        return None, None

    log_limits = _build_log_limits(config.log_limits)
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

    if config.export_logs:
        resolved = _resolve_otlp(config.otlp, config.otlp_logs)
        exporter = _build_otlp_log_exporter(resolved)
        batch_processor_cls = _get_batch_log_record_processor_cls()
        processor_kwargs = _filter_kwargs(
            cast("Callable[..., object]", batch_processor_cls),
            _build_batch_kwargs(config.logs_batch),
        )
        batch_processor_ctor = cast("Callable[..., object]", batch_processor_cls)
        logger_provider.add_log_record_processor(batch_processor_ctor(exporter, **processor_kwargs))

    handler_cls = _get_logging_handler_cls()
    handler_ctor = cast("Callable[..., logging.Handler]", handler_cls)
    log_handler = handler_ctor(level=logging.NOTSET, logger_provider=logger_provider)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    if config.logs_trace_filter:
        log_handler.addFilter(_trace_sampled_log_filter())

    return logger_provider, log_handler


def _instrument_runtime(config: ObservabilityConfig, registry: InstrumentationRegistry) -> None:
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

    if config.log_correlation or config.logs_auto_instrument:
        _instrument("logging", LoggingInstrumentor(), set_logging_format=False)
    else:
        registry.record_suppressed("logging")

    _instrument("threading", ThreadingInstrumentor())
    _instrument("asyncio", AsyncioInstrumentor())
    _instrument("httpx", HTTPXClientInstrumentor())
    _instrument("requests", RequestsInstrumentor())


def _configure_propagators(config: ObservabilityConfig) -> None:
    if not config.propagators:
        return

    propagators: list[TextMapPropagator] = []
    for name in config.propagators:
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


def _apply_config_file(path: Path) -> bool:
    otel_config = _load_module(
        "opentelemetry.sdk._configuration",
        label="OpenTelemetry SDK configuration",
    )

    candidates: list[Callable[..., object]] = []
    for name in ("configure", "configure_otel", "initialize", "init", "load"):
        value = getattr(otel_config, name, None)
        if callable(value):
            candidates.append(value)

    configurator_cls = getattr(otel_config, "Configurator", None)
    if configurator_cls is not None:
        configurator = configurator_cls()
        configure = getattr(configurator, "configure", None)
        if callable(configure):
            candidates.append(configure)

    for candidate in candidates:
        if _call_configurator(candidate, path):
            return True

    LOG.warning("No compatible OpenTelemetry configuration entrypoint found")
    return False


def _call_configurator(func: Callable[..., object], path: Path) -> bool:
    config_value = str(path)
    attempts: tuple[Callable[[], object], ...] = (
        partial(func, config_file=config_value),
        partial(func, config_file_path=config_value),
        partial(func, path=config_value),
        partial(func, config_value),
        partial(func),
    )

    for attempt in attempts:
        try:
            attempt()
        except TypeError:
            continue
        except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover
            LOG.warning("Failed to apply OpenTelemetry config: %s", exc)
            return False
        else:
            return True

    return False


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


def _runtime_from_global(
    config: ObservabilityConfig,
    *,
    log_handler: logging.Handler | None,
    logger_provider: _LoggerProvider | None,
    grpc_handle: GrpcObservabilityHandle | None,
    prometheus_enabled: bool,
) -> ObservabilityRuntime:
    tracer_provider = cast("TracerProvider", otel_trace.get_tracer_provider())
    meter_provider = cast("MeterProvider", otel_metrics.get_meter_provider())
    tracer = otel_trace.get_tracer(config.service_name)
    meter = otel_metrics.get_meter(config.service_name)
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


def _apply_collector_local_config(config: ObservabilityConfig) -> ObservabilityConfig:
    local_otlp = replace(config.otlp, endpoint="http://localhost:4317", protocol="grpc")
    local_traces = replace(config.otlp_traces, endpoint="http://localhost:4317", protocol="grpc")
    local_metrics = replace(config.otlp_metrics, endpoint="http://localhost:4317", protocol="grpc")
    local_logs = replace(config.otlp_logs, endpoint="http://localhost:4317", protocol="grpc")
    return replace(
        config,
        otlp=local_otlp,
        otlp_traces=local_traces,
        otlp_metrics=local_metrics,
        otlp_logs=local_logs,
        config_file=None,
        console_export=False,
        prometheus_enabled=False,
    )


def _runtime_from_in_memory(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
) -> ObservabilityRuntime:
    resource = _build_resource(config)
    span_limits = _build_span_limits(config.span_limits)
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(
        sampler=ALWAYS_ON,
        resource=resource,
        span_limits=span_limits,
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(config.service_name)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
        resource=resource,
        exemplar_filter=_build_exemplar_filter(config),
        views=_build_views(config),
    )
    otel_metrics.set_meter_provider(meter_provider)
    meter = otel_metrics.get_meter(config.service_name)

    _instrument_runtime(config, registry)

    registry.emit_summary(LOG)
    registry.emit_metrics(meter)

    shutdown = _build_shutdown(
        tracer_provider,
        meter_provider,
        logger_provider=None,
        log_handler=None,
        grpc_handle=None,
    )

    return ObservabilityRuntime(
        enabled=True,
        tracer=tracer,
        meter=meter,
        logger_provider=None,
        log_handler=None,
        shutdown=shutdown,
        policy=config.policy,
        prometheus_enabled=False,
        grpc_observability=None,
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
        test_handles=TestTelemetryHandles(
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        ),
    )


def _runtime_from_config_file(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
) -> ObservabilityRuntime | None:
    if not config.config_file:
        return None

    try:
        validate_otel_config_file(config.config_file)
    except (ValueError, FileNotFoundError) as exc:
        message = f"Invalid OpenTelemetry config file: {exc}"
        raise RuntimeError(message) from exc

    configured = _apply_config_file(config.config_file)
    if not configured:
        LOG.warning(
            "Failed to apply OTEL config file %s; using manual config",
            config.config_file,
        )
        return None

    _instrument_runtime(config, registry)
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
            if config.logs_trace_filter:
                log_handler.addFilter(_trace_sampled_log_filter())

    grpc_handle = register_grpc_observability(
        config.grpc_observability,
        meter_provider=otel_metrics.get_meter_provider(),
        registry=registry,
    )
    registry.emit_summary(LOG)
    registry.emit_metrics(otel_metrics.get_meter(config.service_name))
    return _runtime_from_global(
        config,
        log_handler=log_handler,
        logger_provider=logger_provider,
        grpc_handle=grpc_handle,
        prometheus_enabled=config.prometheus_enabled,
    )


def _runtime_from_manual_config(
    config: ObservabilityConfig,
    registry: InstrumentationRegistry,
) -> ObservabilityRuntime:
    resource = _build_resource(config)

    tracer_provider = _build_tracer_provider(config, resource)
    otel_trace.set_tracer_provider(tracer_provider)
    tracer = otel_trace.get_tracer(config.service_name)

    meter_provider, prometheus_enabled = _build_meter_provider(config, resource)
    otel_metrics.set_meter_provider(meter_provider)
    meter = otel_metrics.get_meter(config.service_name)

    logger_provider, log_handler = _build_logger_provider(config, resource)
    if logger_provider is not None:
        logs_api = _load_otel_logs_api()
        set_logger_provider = getattr(logs_api, "set_logger_provider", None)
        if callable(set_logger_provider):
            set_logger_provider(logger_provider)

    _instrument_runtime(config, registry)

    grpc_handle = register_grpc_observability(
        config.grpc_observability,
        meter_provider=meter_provider,
        registry=registry,
    )

    registry.emit_summary(LOG)
    registry.emit_metrics(meter)

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


def _init_observability(config: ObservabilityConfig) -> ObservabilityRuntime:
    test_mode = config.test_mode
    if test_mode == "disabled":
        return _disabled_runtime()

    if config.config_file is None:
        _configure_propagators(config)

    registry = get_instrumentation_registry()
    if test_mode == "in_memory":
        return _runtime_from_in_memory(config, registry)

    if test_mode == "collector_local":
        config = _apply_collector_local_config(config)

    if not config.enabled:
        return _disabled_runtime()

    runtime = _runtime_from_config_file(config, registry)
    if runtime is not None:
        return runtime
    return _runtime_from_manual_config(config, registry)


def bootstrap_observability(config: ObservabilityConfig) -> ObservabilityRuntime:
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
    """Return the global observability runtime manager."""
    return _RUNTIME_MANAGER


def _log_shutdown_result(result: ObservabilityShutdownResult) -> None:
    payload = json.dumps(result.to_log_payload(), sort_keys=True)
    if result.flush_ok:
        LOG.info("telemetry.flush %s", payload)
    else:
        LOG.warning("telemetry.flush %s", payload)


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
    attrs = shape_attributes(
        {"action": action},
        allowed_keys=frozenset({"action"}),
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
    """Return the last telemetry pipeline flush summary."""
    return PipelineHealthState(
        last_flush_ok=_PIPELINE_HEALTH_STATE.last_flush_ok,
        last_flush_ms=_PIPELINE_HEALTH_STATE.last_flush_ms,
        last_flush_errors=_PIPELINE_HEALTH_STATE.last_flush_errors,
    )


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
        service_version=settings.service_version,
        deployment_environment=settings.deployment_environment,
        resource_attributes=settings.resource_attributes,
        propagators=settings.propagators,
        traces_sampler=settings.traces_sampler,
        traces_sampler_arg=settings.traces_sampler_arg,
        config_file=settings.config_file,
        otlp=settings.otlp,
        otlp_traces=settings.otlp_traces,
        otlp_metrics=settings.otlp_metrics,
        otlp_logs=settings.otlp_logs,
        export_traces=settings.export_traces,
        export_metrics=settings.export_metrics,
        export_logs=settings.export_logs,
        console_export=settings.console_export,
        prometheus_enabled=settings.prometheus_enabled,
        logs_auto_instrument=settings.logs_auto_instrument,
        log_correlation=settings.log_correlation,
        logs_trace_filter=settings.logs_trace_filter,
        traces_batch=settings.traces_batch,
        logs_batch=settings.logs_batch,
        metrics_export=settings.metrics_export,
        span_limits=settings.span_limits,
        log_limits=settings.log_limits,
        metrics_exemplar_filter=settings.metrics_exemplar_filter,
        metric_views=settings.metric_views,
        grpc_observability=settings.grpc_observability,
        hamilton_tracker=settings.hamilton_tracker,
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
        test_mode=resolve_test_telemetry_mode(),
        policy=policy_from_settings(settings),
    )


__all__ = [
    "ObservabilityConfig",
    "ObservabilityRuntime",
    "ObservabilityShutdownResult",
    "ObservabilityRuntimeManager",
    "PipelineHealthState",
    "TestTelemetryHandles",
    "bootstrap_observability",
    "build_exemplar_filter",
    "build_metric_views",
    "flush_observability",
    "get_pipeline_health_state",
    "get_observability",
    "get_runtime_manager",
    "observability_config_from_settings",
    "shutdown_observability",
]
