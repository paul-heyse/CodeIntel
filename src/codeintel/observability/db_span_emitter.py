"""Shared DB span emission utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.observability.context import get_correlation_id
from codeintel.observability.db_span_attributes import DbSpanAttributeBuilder

if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanKind, Tracer

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind as _SpanKind
    from opentelemetry.trace.status import Status, StatusCode

    _SPAN_KIND_CLIENT: SpanKind | None = _SpanKind.CLIENT
except ImportError:
    otel_trace = None
    Status = None
    StatusCode = None
    _SPAN_KIND_CLIENT = None

SpanAttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True, slots=True)
class DbSpanEmitterConfig:
    """Configuration for emitting DB spans."""

    tracer: Tracer
    db_system_name: str
    db_namespace: str | None
    attributes: Mapping[str, object]
    span_builder: DbSpanAttributeBuilder
    require_parent_span: bool


class DbSpanEmitter:
    """Emit OpenTelemetry spans for database operations."""

    def __init__(self, config: DbSpanEmitterConfig) -> None:
        """Initialize the emitter with a fixed configuration."""
        self._config = config

    def trace_call(
        self,
        *,
        sql: str,
        params: object | None,
        is_batch: bool,
        call: Callable[[], object],
    ) -> object:
        """Trace a database call using the configured span builder.

        Returns
        -------
        object
            Result of the wrapped database call.
        """
        if self._config.require_parent_span and not _has_parent_span():
            return call()

        spec = self._config.span_builder.build(
            sql=sql,
            params=params,
            db_system_name=self._config.db_system_name,
            db_namespace=self._config.db_namespace,
            is_batch=is_batch,
        )

        with _start_span(self._config.tracer, spec.name) as span:
            _set_span_attributes(span, spec.attributes)
            correlation_id = get_correlation_id()
            if correlation_id:
                span.set_attribute("codeintel.correlation_id", correlation_id)
            _set_span_attributes(span, self._config.attributes)
            try:
                return call()
            except Exception as exc:  # pragma: no cover
                _record_span_error(span, exc)
                raise


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


def _start_span(tracer: Tracer, operation: str) -> AbstractContextManager[Span]:
    if _SPAN_KIND_CLIENT is None:
        return tracer.start_as_current_span(operation)
    return tracer.start_as_current_span(operation, kind=_SPAN_KIND_CLIENT)


def _set_span_attributes(span: Span, attrs: Mapping[str, object]) -> None:
    for key, value in attrs.items():
        attr_value = _coerce_attribute_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def _record_span_error(span: Span, exc: Exception) -> None:
    span.record_exception(exc)
    if Status is not None and StatusCode is not None:
        span.set_status(Status(StatusCode.ERROR))


def _has_parent_span() -> bool:
    if otel_trace is None:
        return False
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    return context.is_valid


__all__ = ["DbSpanEmitter", "DbSpanEmitterConfig"]
