"""Telemetry utilities for ingestion plugin execution.

This module provides ingestion-specific telemetry that extends the base
RuntimeTelemetry from core with ingestion-specific span tracking, metrics,
and the OtelIngestRunSink for run-level metrics.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from codeintel.core.execution.telemetry import (
    OTEL_AVAILABLE,
    PluginSpan,
    RuntimeTelemetry,
    TelemetryConfig,
)
from codeintel.core.singleton import SingletonHolder

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


# Alias for backwards compatibility
IngestPluginSpan = PluginSpan


class IngestTelemetryConfig(TelemetryConfig):
    """Configuration for ingestion telemetry."""

    service_name: str = "codeintel.ingestion"


class IngestRuntimeTelemetry(RuntimeTelemetry):
    """Telemetry manager for ingestion plugin execution.

    Extend the base RuntimeTelemetry with ingestion-specific methods
    including row count histograms and specialized span tracking.

    Examples
    --------
    >>> telemetry = get_ingest_telemetry()
    >>> span = telemetry.start_plugin_span(plugin, run_id)
    >>> try:
    ...     result = plugin.execute(ctx)
    ...     telemetry.end_span(span, success=True, rows_written=100)
    ... except Exception:
    ...     telemetry.end_span(span, success=False, error="...")
    """

    def __init__(
        self,
        *,
        service_name: str = "codeintel.ingestion",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize ingestion telemetry.

        Parameters
        ----------
        service_name
            Service name for traces and metrics.
        enable_tracing
            Whether tracing is enabled.
        enable_metrics
            Whether metrics are enabled.
        """
        config = TelemetryConfig(
            service_name=service_name,
            enable_tracing=enable_tracing,
            enable_metrics=enable_metrics,
        )
        super().__init__(config)
        self._rows_histogram: _MetricRecorder | None = None
        self._init_rows_histogram()

    def _init_rows_histogram(self) -> None:
        """Initialize rows histogram for ingestion-specific metrics."""
        if self._meter is not None:
            self._rows_histogram = self._meter.create_histogram(
                "codeintel.ingest.plugin.rows",
                unit="rows",
                description="Rows written by ingestion plugin",
            )

    def start_plugin_span(
        self,
        plugin: IngestPluginProtocol,
        run_id: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> PluginSpan:
        """Start a telemetry span for ingestion plugin execution.

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
        PluginSpan
            Span object for tracking execution.
        """
        merged_attrs = {
            "plugin.name": plugin.metadata.name,
            "plugin.stage": plugin.metadata.stage,
            "plugin.tables": ",".join(plugin.metadata.produces_tables),
            **(attributes or {}),
        }
        return self.start_span(
            plugin.metadata.name,
            run_id,
            attributes=merged_attrs,
        )

    def end_span_with_rows(
        self,
        span: PluginSpan,
        *,
        success: bool,
        rows_written: int = 0,
        error: str | None = None,
    ) -> float:
        """End a telemetry span and record both duration and row metrics.

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
        # Record rows metric
        if self._rows_histogram is not None and success:
            labels = {
                "plugin": span.plugin_name,
                "status": "success",
            }
            self._rows_histogram.record(float(rows_written), labels)

        # Use the base class end_span
        return self.end_span(
            span,
            success=success,
            rows_written=rows_written,
            error=error,
        )


class _IngestTelemetryHolder(SingletonHolder[IngestRuntimeTelemetry]):
    """Singleton holder for IngestRuntimeTelemetry."""


def get_ingest_telemetry() -> IngestRuntimeTelemetry:
    """Get the singleton ingestion telemetry instance.

    Returns
    -------
    IngestRuntimeTelemetry
        Shared telemetry instance.
    """
    return _IngestTelemetryHolder.get(IngestRuntimeTelemetry)


def reset_ingest_telemetry() -> None:
    """Reset the telemetry singleton for testing."""
    _IngestTelemetryHolder.reset()


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
        if not OTEL_AVAILABLE:
            try:
                metrics_module = importlib.import_module("opentelemetry.metrics")
            except ImportError as exc:  # pragma: no cover - optional dependency
                message = "opentelemetry not installed; OtelIngestRunSink cannot emit metrics"
                raise RuntimeError(message) from exc
        else:
            metrics_module = importlib.import_module("opentelemetry.metrics")

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
    "IngestTelemetryConfig",
    "OtelIngestRunSink",
    "get_ingest_telemetry",
    "reset_ingest_telemetry",
]
