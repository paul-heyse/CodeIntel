"""Shared span and metric helpers for CodeIntel operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from opentelemetry import trace as otel_trace

from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.instrument_registry import get_instrument_registry
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.runtime import get_observability
from codeintel.observability.semconv_keys import (
    CODEINTEL_COMPONENT,
    CODEINTEL_ENDPOINT,
    CODEINTEL_OPERATION,
    CODEINTEL_QUERY_ENDPOINT,
    CODEINTEL_QUERY_HASH,
    CODEINTEL_QUERY_ROW_COUNT,
    CODEINTEL_QUERY_SCHEMA_HASH,
    CODEINTEL_QUERY_TRUNCATED,
    CODEINTEL_QUERY_VIEW_ID,
    CODEINTEL_SUCCESS,
)
from codeintel.observability.telemetry_context import current_telemetry_context

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span


class QueryMetricsLike(Protocol):
    """Protocol for query metrics payloads."""

    @property
    def endpoint(self) -> str: ...

    @property
    def duration_ms(self) -> float: ...

    @property
    def row_count(self) -> int: ...

    @property
    def truncated(self) -> bool: ...

    @property
    def view_id(self) -> str | None: ...

    @property
    def query_hash(self) -> str | None: ...

    @property
    def schema_hash(self) -> str | None: ...


@dataclass(slots=True)
class _Instruments:
    op_calls: Counter
    op_duration_ms: Histogram
    query_calls: Counter
    query_duration_ms: Histogram
    query_row_count: Histogram
    query_truncated: Counter


_INSTRUMENT_REGISTRY = get_instrument_registry()


@dataclass(frozen=True, slots=True)
class OperationDescriptor:
    """Descriptor for a CodeIntel operation span and metrics."""

    component: str
    operation: str

    def span_name(self) -> str:
        """Return the canonical span name for the operation.

        Returns
        -------
        str
            Canonical span name.
        """
        return f"{self.component}.{self.operation}"


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith("/v1/semantic/views/"):
        return "/v1/semantic/views/{view_id}"
    if endpoint.startswith("/v1/export/semantic/"):
        return "/v1/export/semantic/{view_id}"
    return endpoint


def _infer_component(endpoint: str) -> str:
    if endpoint.startswith("mcp:"):
        return "mcp"
    if endpoint.startswith("/"):
        return "http"
    return "unknown"


def _get_instruments(meter: Meter) -> _Instruments:
    def _builder(inner_meter: Meter) -> _Instruments:
        return _Instruments(
            op_calls=inner_meter.create_counter(
                "codeintel.operation.calls",
                unit="1",
                description="Count of CodeIntel operations across transports",
            ),
            op_duration_ms=inner_meter.create_histogram(
                "codeintel.operation.duration_ms",
                unit="ms",
                description="Duration of CodeIntel operations (ms)",
            ),
            query_calls=inner_meter.create_counter(
                "codeintel.query.calls",
                unit="1",
                description="Count of semantic queries and exports",
            ),
            query_duration_ms=inner_meter.create_histogram(
                "codeintel.query.duration_ms",
                unit="ms",
                description="Query duration (ms)",
            ),
            query_row_count=inner_meter.create_histogram(
                "codeintel.query.row_count",
                unit="1",
                description="Row counts for query/export results",
            ),
            query_truncated=inner_meter.create_counter(
                "codeintel.query.truncated",
                unit="1",
                description="Count of truncated query/export responses",
            ),
        )

    return _INSTRUMENT_REGISTRY.get_group(meter, "operation_scope", _builder)


@dataclass(frozen=True)
class _SpanContext:
    bundle: dict[str, str]
    component: str
    operation: str
    attributes: dict[str, object] | None


def _apply_span_attributes(
    span: Span,
    *,
    context: _SpanContext,
    policy: ObservabilityPolicy,
) -> None:
    normalizer = build_attribute_normalizer(policy)
    for key, value in context.bundle.items():
        span.set_attribute(key, value)
    span.set_attribute(CODEINTEL_COMPONENT, context.component)
    span.set_attribute(CODEINTEL_OPERATION, context.operation)
    if not context.attributes:
        return
    allowlist = policy.operation_allowlist_for(context.component, context.operation)
    filtered = normalizer.normalize(
        context.attributes,
        allowed_keys=allowlist,
    )
    for key, value in filtered.items():
        span.set_attribute(key, value)


def record_operation_metrics(
    *,
    component: str | None = None,
    operation: str | None = None,
    descriptor: OperationDescriptor | None = None,
    duration_ms: float,
    success: bool,
) -> None:
    """Record operation-level metrics with low-cardinality labels."""
    obs = get_observability()
    if not obs.enabled or obs.meter is None:
        return
    resolved = _resolve_descriptor(descriptor, component=component, operation=operation)
    instruments = _get_instruments(obs.meter)
    bundle = current_telemetry_context().metric_attributes()
    attrs = {
        CODEINTEL_COMPONENT: resolved.component,
        CODEINTEL_OPERATION: resolved.operation,
        CODEINTEL_SUCCESS: bool(success),
    }
    attrs.update(bundle)
    instruments.op_calls.add(1, attributes=attrs)
    instruments.op_duration_ms.record(duration_ms, attributes=attrs)


@contextmanager
def observe_operation(
    *,
    component: str | None = None,
    operation: str | None = None,
    descriptor: OperationDescriptor | None = None,
    attributes: dict[str, object] | None = None,
) -> Iterator[Span | None]:
    """Create a span and record duration metrics for an operation.

    Yields
    ------
    Span | None
        Active span for the operation, or None when tracing is disabled.
    """
    obs = get_observability()
    policy = obs.policy
    bundle = current_telemetry_context().span_attributes()
    resolved = _resolve_descriptor(descriptor, component=component, operation=operation)

    if obs.enabled and obs.tracer is not None:
        span_cm = obs.tracer.start_as_current_span(resolved.span_name())
    else:
        span_cm = nullcontext(None)

    start = time.perf_counter()
    success = False
    span: Span | None = None
    try:
        with span_cm as active_span:
            span = active_span
            if span is not None:
                span_context = _SpanContext(
                    bundle=bundle,
                    component=resolved.component,
                    operation=resolved.operation,
                    attributes=attributes,
                )
                _apply_span_attributes(span, context=span_context, policy=policy)
            yield span
        success = True
    except Exception as exc:
        if span is not None:
            span.record_exception(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_operation_metrics(
            descriptor=resolved,
            duration_ms=duration_ms,
            success=success,
        )


def record_query_metrics(metrics: QueryMetricsLike) -> None:
    """Record query metrics via OpenTelemetry and attach span attributes."""
    obs = get_observability()
    bundle = current_telemetry_context().metric_attributes()
    normalizer = build_attribute_normalizer(obs.policy)

    normalized_endpoint = _normalize_endpoint(metrics.endpoint)
    component = _infer_component(metrics.endpoint)
    attrs = {
        CODEINTEL_ENDPOINT: normalized_endpoint,
        CODEINTEL_COMPONENT: component,
    }
    attrs.update(bundle)

    if obs.enabled and obs.meter is not None:
        instruments = _get_instruments(obs.meter)
        instruments.query_calls.add(1, attributes=attrs)
        instruments.query_duration_ms.record(metrics.duration_ms, attributes=attrs)
        instruments.query_row_count.record(metrics.row_count, attributes=attrs)
        if metrics.truncated:
            instruments.query_truncated.add(1, attributes=attrs)

    if not obs.enabled:
        return

    span = otel_trace.get_current_span()
    if span is None:
        return

    span_attrs: dict[str, object] = {
        CODEINTEL_QUERY_ENDPOINT: normalized_endpoint,
        CODEINTEL_QUERY_ROW_COUNT: metrics.row_count,
        CODEINTEL_QUERY_TRUNCATED: bool(metrics.truncated),
    }
    if metrics.view_id:
        span_attrs[CODEINTEL_QUERY_VIEW_ID] = metrics.view_id
    if metrics.query_hash:
        span_attrs[CODEINTEL_QUERY_HASH] = metrics.query_hash
    if metrics.schema_hash:
        span_attrs[CODEINTEL_QUERY_SCHEMA_HASH] = metrics.schema_hash
    for key, value in normalizer.normalize(span_attrs).items():
        span.set_attribute(key, value)


def _resolve_descriptor(
    descriptor: OperationDescriptor | None,
    *,
    component: str | None,
    operation: str | None,
) -> OperationDescriptor:
    if descriptor is not None:
        return descriptor
    if not component or not operation:
        message = "OperationDescriptor requires component and operation"
        raise ValueError(message)
    return OperationDescriptor(component=component, operation=operation)


__all__ = [
    "OperationDescriptor",
    "observe_operation",
    "record_operation_metrics",
    "record_query_metrics",
]
