"""Shared telemetry event helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from codeintel.observability.attribute_sanitizer import SpanAttributeValue, coerce_attribute_value

if TYPE_CHECKING:
    from opentelemetry.trace import Span

LOG = logging.getLogger(__name__)


class TelemetryEvent(Protocol):
    """Protocol for telemetry event payloads."""

    def span_attributes(self) -> Mapping[str, SpanAttributeValue]:
        """Return span attributes for telemetry."""

    def event_attributes(self) -> Mapping[str, SpanAttributeValue]:
        """Return event attributes for telemetry."""

    def log_payload(self) -> Mapping[str, object]:
        """Return log payload fields for structured logging."""


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


def emit_event_log(event: TelemetryEvent, *, event_name: str) -> None:
    """Emit a structured event payload to the logger."""
    payload = json.dumps(
        {
            "event": event_name,
            **event.log_payload(),
        },
        sort_keys=True,
    )
    LOG.info("%s %s", event_name, payload)


__all__ = ["TelemetryEvent", "add_span_event", "emit_event_log", "set_span_attributes"]
