"""Metrics middleware for ingestion plugin execution.

This module provides metrics collection for plugin execution,
including timing histograms and row count counters.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.ingestion.core.base import BaseIngestPlugin
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.plugins.protocol import IngestPluginResult

log = logging.getLogger(__name__)


class MetricRecorder(Protocol):
    """Protocol for OpenTelemetry-style metric recorders."""

    def record(self, value: float, attributes: Mapping[str, str]) -> None:
        """Record a measurement.

        Parameters
        ----------
        value
            Measurement value.
        attributes
            Labels/attributes for the measurement.
        """
        ...


class CounterRecorder(Protocol):
    """Protocol for OpenTelemetry-style counter recorders."""

    def add(self, value: int, attributes: Mapping[str, str]) -> None:
        """Add to a counter.

        Parameters
        ----------
        value
            Value to add.
        attributes
            Labels/attributes for the measurement.
        """
        ...


@dataclass
class InMemoryMetrics:
    """In-memory metrics collector for testing and simple deployments.

    Collect metrics in memory for inspection or export.
    """

    durations: list[tuple[float, Mapping[str, str]]] = field(default_factory=list)
    row_counts: list[tuple[int, Mapping[str, str]]] = field(default_factory=list)
    error_counts: list[tuple[int, Mapping[str, str]]] = field(default_factory=list)

    def record_duration(self, value: float, attributes: Mapping[str, str]) -> None:
        """Record a duration measurement.

        Parameters
        ----------
        value
            Duration in seconds.
        attributes
            Metric labels.
        """
        self.durations.append((value, dict(attributes)))

    def record_rows(self, value: int, attributes: Mapping[str, str]) -> None:
        """Record a row count.

        Parameters
        ----------
        value
            Number of rows.
        attributes
            Metric labels.
        """
        self.row_counts.append((value, dict(attributes)))

    def record_error(self, value: int, attributes: Mapping[str, str]) -> None:
        """Record an error count.

        Parameters
        ----------
        value
            Error count (usually 1).
        attributes
            Metric labels.
        """
        self.error_counts.append((value, dict(attributes)))

    def clear(self) -> None:
        """Clear all collected metrics."""
        self.durations.clear()
        self.row_counts.clear()
        self.error_counts.clear()


@dataclass
class MetricsMiddleware:
    """Middleware that collects plugin execution metrics.

    Collect timing and row count metrics for plugin execution.
    Supports both OpenTelemetry recorders and in-memory collection.

    Attributes
    ----------
    duration_recorder
        Optional histogram recorder for durations.
    rows_recorder
        Optional counter recorder for row counts.
    error_recorder
        Optional counter recorder for errors.
    in_memory
        Optional in-memory metrics collector.
    metric_prefix
        Prefix for metric names.
    """

    duration_recorder: MetricRecorder | None = None
    rows_recorder: CounterRecorder | None = None
    error_recorder: CounterRecorder | None = None
    in_memory: InMemoryMetrics | None = None
    metric_prefix: str = "codeintel.ingest"
    _start_times: dict[str, float] = field(default_factory=dict, repr=False)

    def before_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Record execution start time.

        Parameters
        ----------
        plugin
            The plugin about to execute.
        ctx
            Execution context (unused, required by protocol).
        """
        plugin_name = plugin.metadata.name
        self._start_times[plugin_name] = time.perf_counter()

    def after_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """Record execution metrics.

        Parameters
        ----------
        plugin
            The plugin that executed.
        ctx
            Execution context.
        result
            Execution result.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration = time.perf_counter() - start_time if start_time else 0.0

        status = "skipped" if result.skipped else ("success" if result.success else "error")
        attributes: dict[str, str] = {
            "plugin": plugin_name,
            "stage": plugin.metadata.stage,
            "status": status,
            "repo": ctx.repo,
        }

        # Record duration
        if self.duration_recorder is not None:
            self.duration_recorder.record(duration, attributes)
        if self.in_memory is not None:
            self.in_memory.record_duration(duration, attributes)

        # Record row counts
        if result.row_counts and not result.skipped:
            total_rows = sum(result.row_counts.values())
            if self.rows_recorder is not None:
                self.rows_recorder.add(total_rows, attributes)
            if self.in_memory is not None:
                self.in_memory.record_rows(total_rows, attributes)

    def on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """Record error metrics.

        Parameters
        ----------
        plugin
            The plugin that failed.
        ctx
            Execution context.
        error
            The exception that was raised.
        """
        plugin_name = plugin.metadata.name
        start_time = self._start_times.pop(plugin_name, None)
        duration = time.perf_counter() - start_time if start_time else 0.0

        attributes: dict[str, str] = {
            "plugin": plugin_name,
            "stage": plugin.metadata.stage,
            "status": "error",
            "error_type": type(error).__name__,
            "repo": ctx.repo,
        }

        # Record duration
        if self.duration_recorder is not None:
            self.duration_recorder.record(duration, attributes)
        if self.in_memory is not None:
            self.in_memory.record_duration(duration, attributes)

        # Record error
        if self.error_recorder is not None:
            self.error_recorder.add(1, attributes)
        if self.in_memory is not None:
            self.in_memory.record_error(1, attributes)


__all__ = [
    "CounterRecorder",
    "InMemoryMetrics",
    "MetricRecorder",
    "MetricsMiddleware",
]
