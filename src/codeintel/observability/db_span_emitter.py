"""Shared DB span emission utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace import SpanKind as _SpanKind
from opentelemetry.trace.status import Status, StatusCode

from codeintel.observability.attribute_sanitizer import coerce_attribute_value, shape_attributes
from codeintel.observability.db_tracing import DbSpanAttributeBuilder
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.telemetry_context import current_telemetry_context

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

_SPAN_KIND_CLIENT: SpanKind | None = _SpanKind.CLIENT


@dataclass(frozen=True, slots=True)
class DbSpanEmitterConfig:
    """Configuration for emitting DB spans."""

    tracer: Tracer
    db_system_name: str
    db_namespace: str | None
    attributes: Mapping[str, object]
    span_builder: DbSpanAttributeBuilder
    require_parent_span: bool
    policy: ObservabilityPolicy


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
            _set_span_attributes(span, current_telemetry_context().span_attributes())
            extra_attrs = shape_attributes(
                self._config.attributes,
                allowed_prefixes=self._config.policy.db_attribute_prefixes,
                budget=self._config.policy.budget,
            )
            _set_span_attributes(span, extra_attrs)
            try:
                return call()
            except Exception as exc:  # pragma: no cover
                _record_span_error(span, exc)
                raise


def _start_span(tracer: Tracer, operation: str) -> AbstractContextManager[Span]:
    if _SPAN_KIND_CLIENT is None:
        return tracer.start_as_current_span(operation)
    return tracer.start_as_current_span(operation, kind=_SPAN_KIND_CLIENT)


def _set_span_attributes(span: Span, attrs: Mapping[str, object]) -> None:
    for key, value in attrs.items():
        attr_value = coerce_attribute_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def _record_span_error(span: Span, exc: Exception) -> None:
    span.record_exception(exc)
    if Status is not None and StatusCode is not None:
        span.set_status(Status(StatusCode.ERROR))


def _has_parent_span() -> bool:
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    return context.is_valid


__all__ = ["DbSpanEmitter", "DbSpanEmitterConfig"]
