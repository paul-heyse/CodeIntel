"""Shared span and metric helpers for CodeIntel operations."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from weakref import WeakKeyDictionary

from codeintel.observability.context import get_correlation_id
from codeintel.observability.otel import get_observability

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span

try:
    from opentelemetry import trace as otel_trace
except ImportError:
    otel_trace = None

SpanAttributeValue = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)


class QueryMetricsLike(Protocol):
    """Protocol for query metrics payloads."""

    @property
    def endpoint(self) -> str:
        """Return the endpoint identifier."""
        ...

    @property
    def duration_ms(self) -> float:
        """Return the query duration in milliseconds."""
        ...

    @property
    def row_count(self) -> int:
        """Return the row count."""
        ...

    @property
    def truncated(self) -> bool:
        """Return whether the response was truncated."""
        ...

    @property
    def view_id(self) -> str | None:
        """Return the view identifier."""
        ...

    @property
    def query_hash(self) -> str | None:
        """Return the query hash."""
        ...

    @property
    def schema_hash(self) -> str | None:
        """Return the schema hash."""
        ...


@dataclass(slots=True)
class _Instruments:
    op_calls: Counter
    op_duration_ms: Histogram
    query_calls: Counter
    query_duration_ms: Histogram
    query_row_count: Histogram
    query_truncated: Counter


_INSTRUMENTS: WeakKeyDictionary[Meter, _Instruments] = WeakKeyDictionary()


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


def _get_instruments() -> _Instruments | None:
    obs = get_observability()
    if not obs.enabled or obs.meter is None:
        return None

    meter: Meter = obs.meter
    instruments = _INSTRUMENTS.get(meter)
    if instruments is not None:
        return instruments

    instruments = _Instruments(
        op_calls=meter.create_counter(
            "codeintel.operation.calls",
            unit="1",
            description="Count of CodeIntel operations across transports",
        ),
        op_duration_ms=meter.create_histogram(
            "codeintel.operation.duration_ms",
            unit="ms",
            description="Duration of CodeIntel operations (ms)",
        ),
        query_calls=meter.create_counter(
            "codeintel.query.calls",
            unit="1",
            description="Count of semantic queries and exports",
        ),
        query_duration_ms=meter.create_histogram(
            "codeintel.query.duration_ms",
            unit="ms",
            description="Query duration (ms)",
        ),
        query_row_count=meter.create_histogram(
            "codeintel.query.row_count",
            unit="1",
            description="Row counts for query/export results",
        ),
        query_truncated=meter.create_counter(
            "codeintel.query.truncated",
            unit="1",
            description="Count of truncated query/export responses",
        ),
    )
    _INSTRUMENTS[meter] = instruments
    return instruments


def _coerce_attribute_value(value: object) -> SpanAttributeValue | None:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, bool, int, float)) for item in value):
            return list(value)
        return str(value)
    return str(value)


def _apply_span_attributes(
    span: Span,
    *,
    correlation_id: str | None,
    component: str,
    operation: str,
    attributes: dict[str, object] | None,
) -> None:
    if correlation_id:
        span.set_attribute("codeintel.correlation_id", correlation_id)
    span.set_attribute("codeintel.component", component)
    span.set_attribute("codeintel.operation", operation)
    if not attributes:
        return
    for key, value in attributes.items():
        attr_value = _coerce_attribute_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def record_operation_metrics(
    *,
    component: str,
    operation: str,
    duration_ms: float,
    success: bool,
) -> None:
    """Record operation-level metrics with low-cardinality labels."""
    instruments = _get_instruments()
    if instruments is None:
        return
    attrs = {
        "codeintel.component": component,
        "codeintel.operation": operation,
        "codeintel.success": bool(success),
    }
    instruments.op_calls.add(1, attributes=attrs)
    instruments.op_duration_ms.record(duration_ms, attributes=attrs)


@contextmanager
def observe_operation(
    *,
    component: str,
    operation: str,
    attributes: dict[str, object] | None = None,
) -> Iterator[Span | None]:
    """Create a span and record duration metrics for an operation.

    Yields
    ------
    Span | None
        Active span when tracing is enabled, otherwise ``None``.
    """
    obs = get_observability()
    cid = get_correlation_id()

    if obs.enabled and obs.tracer is not None:
        span_cm = obs.tracer.start_as_current_span(f"{component}.{operation}")
    else:
        span_cm = nullcontext(None)

    start = time.perf_counter()
    success = False
    span: Span | None = None
    try:
        with span_cm as active_span:
            span = active_span
            if span is not None:
                _apply_span_attributes(
                    span,
                    correlation_id=cid,
                    component=component,
                    operation=operation,
                    attributes=attributes,
                )
            yield span
        success = True
    except Exception as exc:
        if span is not None:
            span.record_exception(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_operation_metrics(
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
        )


def record_query_metrics(metrics: QueryMetricsLike) -> None:
    """Record query metrics via OpenTelemetry and attach span attributes."""
    instruments = _get_instruments()
    obs = get_observability()

    normalized_endpoint = _normalize_endpoint(metrics.endpoint)
    component = _infer_component(metrics.endpoint)
    attrs = {
        "codeintel.endpoint": normalized_endpoint,
        "codeintel.component": component,
    }

    if instruments is not None:
        instruments.query_calls.add(1, attributes=attrs)
        instruments.query_duration_ms.record(metrics.duration_ms, attributes=attrs)
        instruments.query_row_count.record(metrics.row_count, attributes=attrs)
        if metrics.truncated:
            instruments.query_truncated.add(1, attributes=attrs)

    if not obs.enabled or otel_trace is None:
        return

    span = otel_trace.get_current_span()
    if span is None:
        return

    span.set_attribute("codeintel.query.endpoint", normalized_endpoint)
    span.set_attribute("codeintel.query.row_count", metrics.row_count)
    span.set_attribute("codeintel.query.truncated", bool(metrics.truncated))
    if metrics.view_id:
        span.set_attribute("codeintel.query.view_id", metrics.view_id)
    if metrics.query_hash:
        span.set_attribute("codeintel.query.hash", metrics.query_hash)
    if metrics.schema_hash:
        span.set_attribute("codeintel.query.schema_hash", metrics.schema_hash)


__all__ = [
    "observe_operation",
    "record_operation_metrics",
    "record_query_metrics",
]
