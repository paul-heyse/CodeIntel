"""Observability protocol definitions.

This module defines the core protocols for observability,
including metrics, logging, and tracing.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class MetricsProtocol(Protocol):
    """Protocol for metrics collection.

    Metrics implementations record counters, gauges, and histograms
    for observability purposes.

    Examples
    --------
    >>> class PrometheusMetrics:
    ...     def increment(self, name: str, value: float = 1, **labels: str) -> None:
    ...         self._counters[name].inc(value)
    ...
    ...     def gauge(self, name: str, value: float, **labels: str) -> None:
    ...         self._gauges[name].set(value)
    """

    def increment(self, name: str, value: float = 1, **labels: str) -> None:
        """Increment a counter metric.

        Parameters
        ----------
        name
            Metric name.
        value
            Value to increment by.
        **labels
            Metric labels.
        """
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge metric value.

        Parameters
        ----------
        name
            Metric name.
        value
            Current value.
        **labels
            Metric labels.
        """
        ...

    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation.

        Parameters
        ----------
        name
            Metric name.
        value
            Observed value.
        **labels
            Metric labels.
        """
        ...


@runtime_checkable
class TracingProtocol(Protocol):
    """Protocol for distributed tracing.

    Tracing implementations create spans for tracking
    request flow across services.
    """

    def start_span(self, name: str, **attributes: object) -> SpanProtocol:
        """Start a new trace span.

        Parameters
        ----------
        name
            Span name.
        **attributes
            Span attributes.

        Returns
        -------
        SpanProtocol
            The started span.
        """
        ...


@runtime_checkable
class SpanProtocol(Protocol):
    """Protocol for trace spans."""

    def set_attribute(self, key: str, value: object) -> None:
        """Set a span attribute.

        Parameters
        ----------
        key
            Attribute key.
        value
            Attribute value.
        """
        ...

    def set_status(self, status: str, message: str | None = None) -> None:
        """Set the span status.

        Parameters
        ----------
        status
            Status code (ok, error).
        message
            Optional status message.
        """
        ...

    def end(self) -> None:
        """End the span."""
        ...


@runtime_checkable
class ObservabilityProtocol(Protocol):
    """Combined protocol for full observability stack.

    Attributes
    ----------
    COMPONENT_NAME
        Component name for logs and metrics.
    """

    COMPONENT_NAME: ClassVar[str]

    @property
    def metrics(self) -> MetricsProtocol:
        """Get the metrics collector.

        Returns
        -------
        MetricsProtocol
            Metrics instance.
        """
        ...

    @property
    def tracer(self) -> TracingProtocol:
        """Get the tracer.

        Returns
        -------
        TracingProtocol
            Tracer instance.
        """
        ...


__all__ = [
    "MetricsProtocol",
    "ObservabilityProtocol",
    "SpanProtocol",
    "TracingProtocol",
]
