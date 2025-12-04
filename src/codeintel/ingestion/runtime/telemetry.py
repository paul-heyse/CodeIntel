"""Telemetry utilities for ingestion plugin execution.

This module provides telemetry integration for ingestion plugin execution,
including metric recording and span management via OpenTelemetry (when available).
Analogous to graphs/runtime/telemetry.py for structural alignment.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.ingestion.core.runs import IngestRun
    from codeintel.ingestion.plugins.protocol import IngestPluginProtocol

log = logging.getLogger(__name__)


class _MetricRecorder(Protocol):
    """Minimal interface for OpenTelemetry histogram recorders."""

    def record(self, value: float, attributes: Mapping[str, str]) -> None:
        """Record a single measurement."""
        ...


@dataclass
class IngestPluginSpan:
    """Represent a telemetry span for plugin execution.

    Attributes
    ----------
    plugin_name
        Name of the plugin.
    run_id
        Run identifier.
    start_time_ns
        Start time in nanoseconds (monotonic).
    attributes
        Span attributes.
    context_data
        Additional context data.
    """

    plugin_name: str
    run_id: str
    start_time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    context_data: dict[str, Any] = field(default_factory=dict)


class IngestRuntimeTelemetry:
    """Telemetry manager for ingestion plugin execution.

    Handle span creation, metric recording, and integration with
    OpenTelemetry (when available).
    """

    def __init__(
        self,
        *,
        service_name: str = "codeintel.ingestion",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize telemetry.

        Parameters
        ----------
        service_name
            Service name for traces.
        enable_tracing
            Whether tracing is enabled.
        enable_metrics
            Whether metrics are enabled.
        """
        self._service_name = service_name
        self._enable_tracing = enable_tracing
        self._enable_metrics = enable_metrics
        self._tracer: Any | None = None
        self._meter: Any | None = None
        self._duration_histogram: _MetricRecorder | None = None
        self._rows_histogram: _MetricRecorder | None = None
        self._initialize_otel()

    def _initialize_otel(self) -> None:
        """Initialize OpenTelemetry if available."""
        try:
            trace_module = importlib.import_module("opentelemetry.trace")
            metrics_module = importlib.import_module("opentelemetry.metrics")

            if self._enable_tracing:
                self._tracer = trace_module.get_tracer(self._service_name)

            if self._enable_metrics:
                meter = metrics_module.get_meter(self._service_name)
                self._meter = meter
                self._duration_histogram = meter.create_histogram(
                    "codeintel.ingest.plugin.duration",
                    unit="s",
                    description="Ingestion plugin execution duration in seconds",
                )
                self._rows_histogram = meter.create_histogram(
                    "codeintel.ingest.plugin.rows",
                    unit="rows",
                    description="Rows written by ingestion plugin",
                )

            log.debug("OpenTelemetry initialized for ingestion telemetry")
        except ImportError:
            log.debug("OpenTelemetry not available; telemetry will be no-op")
            self._tracer = None
            self._meter = None

    def start_span(
        self,
        plugin: IngestPluginProtocol,
        run_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> IngestPluginSpan:
        """Start a telemetry span for plugin execution.

        Parameters
        ----------
        plugin
            Plugin being executed.
        run_id
            Run identifier.
        attributes
            Optional span attributes.

        Returns
        -------
        IngestPluginSpan
            Span object for tracking execution.
        """
        # Build context data from enabled telemetry features
        context_data: dict[str, Any] = {
            "tracing_enabled": self.tracing_enabled,
            "metrics_enabled": self.metrics_enabled,
        }
        return IngestPluginSpan(
            plugin_name=plugin.metadata.name,
            run_id=run_id,
            start_time_ns=time.perf_counter_ns(),
            attributes=attributes or {},
            context_data=context_data,
        )

    def end_span(
        self,
        span: IngestPluginSpan,
        *,
        success: bool,
        rows_written: int = 0,
        error: str | None = None,
    ) -> float:
        """End a telemetry span and record metrics.

        Parameters
        ----------
        span
            Span to end.
        success
            Whether execution succeeded.
        rows_written
            Number of rows written.
        error
            Error message if failed.

        Returns
        -------
        float
            Duration in seconds.
        """
        duration_ns = time.perf_counter_ns() - span.start_time_ns
        duration_s = duration_ns / 1e9

        labels = {
            "plugin": span.plugin_name,
            "status": "success" if success else "error",
        }
        if error:
            labels["error_type"] = error

        if self._duration_histogram:
            self._duration_histogram.record(duration_s, labels)

        if self._rows_histogram and success:
            self._rows_histogram.record(float(rows_written), labels)

        if error:
            log.debug(
                "Plugin span ended: plugin=%s duration=%.3fs success=%s error=%s",
                span.plugin_name,
                duration_s,
                success,
                error,
            )
        else:
            log.debug(
                "Plugin span ended: plugin=%s duration=%.3fs success=%s rows=%d",
                span.plugin_name,
                duration_s,
                success,
                rows_written,
            )

        return duration_s

    @property
    def tracing_enabled(self) -> bool:
        """Return True if tracing is enabled and available."""
        return self._tracer is not None

    @property
    def metrics_enabled(self) -> bool:
        """Return True if metrics are enabled and available."""
        return self._meter is not None


@lru_cache(maxsize=1)
def get_ingest_telemetry() -> IngestRuntimeTelemetry:
    """Get the singleton ingestion telemetry instance.

    Returns
    -------
    IngestRuntimeTelemetry
        Shared telemetry instance.
    """
    return IngestRuntimeTelemetry()


@dataclass
class OtelIngestRunSink:
    """Sink that emits IngestRun metrics via OpenTelemetry.

    This sink integrates with the run tracking system to emit
    metrics for each ingestion run.
    """

    meter_name: str = "codeintel.ingestion"
    _duration: _MetricRecorder = field(init=False, repr=False)
    _rows_inserted: _MetricRecorder = field(init=False, repr=False)
    _rows_deleted: _MetricRecorder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize OpenTelemetry recorders if the dependency is installed.

        Raises
        ------
        RuntimeError
            If the optional opentelemetry dependency is missing.
        """
        try:
            metrics_module = importlib.import_module("opentelemetry.metrics")
        except ImportError as exc:  # pragma: no cover - optional dependency
            message = "opentelemetry not installed; OtelIngestRunSink cannot emit metrics"
            raise RuntimeError(message) from exc
        meter = metrics_module.get_meter(self.meter_name)
        self._duration = cast(
            "_MetricRecorder",
            meter.create_histogram(
                "codeintel.ingest.duration",
                unit="s",
                description="Ingestion step duration in seconds",
            ),
        )
        self._rows_inserted = cast(
            "_MetricRecorder",
            meter.create_histogram(
                "codeintel.ingest.rows_inserted",
                unit="rows",
                description="Rows inserted by an ingestion step",
            ),
        )
        self._rows_deleted = cast(
            "_MetricRecorder",
            meter.create_histogram(
                "codeintel.ingest.rows_deleted",
                unit="rows",
                description="Rows deleted by an ingestion step",
            ),
        )

    def record(self, run: IngestRun) -> None:
        """Emit metrics for the provided run.

        Parameters
        ----------
        run
            Ingestion run to record metrics for.
        """
        labels = {
            "repo": run.repo,
            "step": run.step,
            "status": run.status.value,
            "mode": run.mode.value,
        }
        if run.duration_s is not None:
            self._duration.record(run.duration_s, labels)
        self._rows_inserted.record(float(run.rows_inserted), labels)
        self._rows_deleted.record(float(run.rows_deleted), labels)


__all__ = [
    "IngestPluginSpan",
    "IngestRuntimeTelemetry",
    "OtelIngestRunSink",
    "get_ingest_telemetry",
]
