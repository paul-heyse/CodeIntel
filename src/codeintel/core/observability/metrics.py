"""Metrics collection utilities.

This module provides utilities for collecting and reporting metrics.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric value with labels.

    Attributes
    ----------
    name
        Metric name.
    value
        Metric value.
    labels
        Metric labels.
    timestamp
        When the metric was recorded.
    """

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class InMemoryMetrics:
    """In-memory metrics collector for testing and development.

    Stores metrics in memory for later inspection.

    Examples
    --------
    >>> metrics = InMemoryMetrics()
    >>> metrics.increment("requests_total", 1, method="GET")
    >>> metrics.get_counter("requests_total")
    1.0
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._history: list[MetricValue] = []

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
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        self._history.append(MetricValue(name, value, labels))

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
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._history.append(MetricValue(name, value, labels))

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
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        self._history.append(MetricValue(name, value, labels))

    def get_counter(self, name: str, **labels: str) -> float:
        """Get current counter value.

        Parameters
        ----------
        name
            Metric name.
        **labels
            Metric labels.

        Returns
        -------
        float
            Current counter value.
        """
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, **labels: str) -> float | None:
        """Get current gauge value.

        Parameters
        ----------
        name
            Metric name.
        **labels
            Metric labels.

        Returns
        -------
        float | None
            Current gauge value or None.
        """
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram(self, name: str, **labels: str) -> list[float]:
        """Get histogram observations.

        Parameters
        ----------
        name
            Metric name.
        **labels
            Metric labels.

        Returns
        -------
        list[float]
            List of observed values.
        """
        key = self._make_key(name, labels)
        return list(self._histograms.get(key, []))

    def clear(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._history.clear()

    @staticmethod
    def _make_key(name: str, labels: dict[str, str]) -> str:
        """Create a unique key for a metric.

        Parameters
        ----------
        name
            Metric name.
        labels
            Metric labels.

        Returns
        -------
        str
            Unique key.
        """
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


@contextmanager
def timed_metric(
    metrics: InMemoryMetrics,
    name: str,
    **labels: str,
) -> Iterator[None]:
    """Context manager for timing operations.

    Parameters
    ----------
    metrics
        Metrics collector.
    name
        Histogram metric name.
    **labels
        Metric labels.

    Yields
    ------
    None
        Context for the timed operation.

    Examples
    --------
    >>> with timed_metric(metrics, "request_duration_seconds", method="GET"):
    ...     process_request()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        metrics.histogram(name, duration, **labels)


__all__ = [
    "InMemoryMetrics",
    "MetricValue",
    "timed_metric",
]
