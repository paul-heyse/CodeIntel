"""Shared telemetry event helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.observability.attribute_sanitizer import SpanAttributeValue, coerce_attribute_value
from codeintel.observability.attribute_schema import AttributeNormalizer
from codeintel.observability.telemetry_context import TelemetryContext, current_telemetry_context

if TYPE_CHECKING:
    from opentelemetry.trace import Span

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Structured telemetry event payload."""

    name: str
    span_attributes: Mapping[str, SpanAttributeValue] = field(default_factory=dict)
    event_attributes: Mapping[str, SpanAttributeValue] = field(default_factory=dict)
    log_payload: Mapping[str, object] = field(default_factory=dict)
    span_event_name: str | None = None
    log_event_name: str | None = None
    log_level: int = logging.INFO


def set_span_attributes(span: Span, attributes: Mapping[str, SpanAttributeValue]) -> None:
    """Apply attributes to a span with safe coercion."""
    for key, value in attributes.items():
        attr_value = coerce_attribute_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def add_span_event(
    span: Span,
    name: str,
    attributes: Mapping[str, SpanAttributeValue],
) -> None:
    """Add a span event with sanitized attributes."""
    event_attrs: dict[str, SpanAttributeValue] = {}
    for key, value in attributes.items():
        attr_value = coerce_attribute_value(value)
        if attr_value is not None:
            event_attrs[key] = attr_value
    if event_attrs:
        span.add_event(name, attributes=event_attrs)


def emit_event_log(
    payload: Mapping[str, object],
    *,
    event_name: str,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a structured event payload to the logger."""
    message = json.dumps(
        {
            "event": event_name,
            **payload,
        },
        sort_keys=True,
    )
    target = logger or LOG
    target.log(level, "%s %s", event_name, message)


def emit_event(
    *,
    event: TelemetryEvent,
    span: Span | None,
    normalizer: AttributeNormalizer,
    logger: logging.Logger | None = None,
    context: TelemetryContext | None = None,
) -> None:
    """Emit a telemetry event to logs and span events."""
    resolved_context = context or current_telemetry_context()
    span_event_name = event.span_event_name or event.name
    log_event_name = event.log_event_name or event.name

    span_attrs = {**resolved_context.span_attributes(), **event.span_attributes}
    normalized_span = normalizer.normalize(span_attrs)
    if span is not None:
        set_span_attributes(span, normalized_span)
        event_attrs = normalizer.normalize(event.event_attributes)
        if event_attrs:
            add_span_event(span, span_event_name, event_attrs)

    payload = {**event.log_payload, **resolved_context.span_attributes()}
    emit_event_log(
        payload,
        event_name=log_event_name,
        level=event.log_level,
        logger=logger,
    )


__all__ = [
    "TelemetryEvent",
    "add_span_event",
    "emit_event",
    "emit_event_log",
    "set_span_attributes",
]
