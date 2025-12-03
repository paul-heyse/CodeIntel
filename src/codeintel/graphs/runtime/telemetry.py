"""Telemetry utilities for graph plugin execution.

This module provides telemetry integration for graph plugin execution,
including span management and metric recording without any dependency
on the analytics subsystem.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import StatusCode
except ImportError:  # pragma: no cover - optional dependency
    metrics = None
    trace = None
    StatusCode = None

if TYPE_CHECKING:
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.core.context import GraphExecutionContext
    from codeintel.graphs.core.protocol import GraphPluginProtocol
    from codeintel.graphs.core.result import GraphPluginRunRecord

log = logging.getLogger(__name__)


@dataclass
class GraphPluginSpan:
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


class GraphRuntimeTelemetry:
    """Telemetry manager for graph plugin execution.

    Handle span creation, metric recording, and integration with
    OpenTelemetry (when available).
    """

    def __init__(
        self,
        *,
        service_name: str = "codeintel.graphs",
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
        self._tracing_enabled = enable_tracing
        self._metrics_enabled = enable_metrics
        self._tracer: Any = None
        self._meter: Any = None
        self._plugin_duration_histogram: Any = None
        self._plugin_success_counter: Any = None
        self._plugin_failure_counter: Any = None
        self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry integration if available."""
        if trace is None or metrics is None:
            log.debug("OpenTelemetry not available; telemetry disabled")
            self._tracing_enabled = False
            self._metrics_enabled = False
            return

        if self._tracing_enabled:
            self._tracer = trace.get_tracer(self._service_name)

        if self._metrics_enabled:
            self._meter = metrics.get_meter(self._service_name)
            self._plugin_duration_histogram = self._meter.create_histogram(
                name="graph_plugin_duration_ms",
                description="Duration of graph plugin execution in milliseconds",
                unit="ms",
            )
            self._plugin_success_counter = self._meter.create_counter(
                name="graph_plugin_success_total",
                description="Total successful graph plugin executions",
            )
            self._plugin_failure_counter = self._meter.create_counter(
                name="graph_plugin_failure_total",
                description="Total failed graph plugin executions",
            )

    def start_plugin(
        self,
        plugin: GraphPluginProtocol,
        run_id: str,
        ctx: GraphExecutionContext,
    ) -> GraphPluginSpan:
        """Start a telemetry span for plugin execution.

        Parameters
        ----------
        plugin
            Plugin being executed.
        run_id
            Run identifier.
        ctx
            Execution context.

        Returns
        -------
        GraphPluginSpan
            Span object for tracking execution.
        """
        span = GraphPluginSpan(
            plugin_name=plugin.metadata.name,
            run_id=run_id,
            start_time_ns=time.perf_counter_ns(),
            attributes={
                "plugin.name": plugin.metadata.name,
                "plugin.kind": plugin.metadata.kind,
                "plugin.stage": plugin.metadata.stage,
                "repo": ctx.repo,
                "commit": ctx.commit,
                "run_id": run_id,
            },
        )

        if self._tracer is not None:
            try:
                otel_span = self._tracer.start_span(
                    f"graph.plugin.{plugin.metadata.name}",
                    attributes=span.attributes,
                )
                span.context_data["otel_span"] = otel_span
            except (ImportError, AttributeError):
                pass

        return span

    @staticmethod
    def finish_plugin(
        span: GraphPluginSpan,
        record: GraphPluginRunRecord,
    ) -> None:
        """Finish a telemetry span with execution results.

        Parameters
        ----------
        span
            Span started with start_plugin.
        record
            Execution record with results.
        """
        duration_ns = time.perf_counter_ns() - span.start_time_ns
        duration_ms = duration_ns / 1_000_000

        log.debug(
            "graph_telemetry.plugin.finish name=%s status=%s duration_ms=%.2f",
            span.plugin_name,
            record.status,
            duration_ms,
        )

        otel_span = span.context_data.get("otel_span")
        if otel_span is not None:
            try:
                otel_span.set_attribute("status", record.status)
                otel_span.set_attribute("duration_ms", duration_ms)
                otel_span.set_attribute("attempts", record.attempts)
                if record.error and StatusCode is not None:
                    otel_span.set_attribute("error", record.error)
                    otel_span.set_status(StatusCode.ERROR, record.error)
                elif StatusCode is not None:
                    otel_span.set_status(StatusCode.OK)
                otel_span.end()
            except AttributeError:
                pass

    def record_metrics(
        self,
        record: GraphPluginRunRecord,
        scope: GraphRunScope | None,
    ) -> None:
        """Record execution metrics.

        Parameters
        ----------
        record
            Execution record with results.
        scope
            Execution scope.
        """
        if not self._metrics_enabled:
            return

        labels = {
            "plugin_name": record.name,
            "status": record.status,
        }
        if scope is not None:
            if scope.paths:
                labels["scope_path_count"] = len(scope.paths)
            if scope.modules:
                labels["scope_module_count"] = len(scope.modules)

        try:
            if self._plugin_duration_histogram is not None:
                self._plugin_duration_histogram.record(record.duration_ms, labels)

            if record.status == "succeeded" and self._plugin_success_counter is not None:
                self._plugin_success_counter.add(1, labels)
            elif record.status == "failed" and self._plugin_failure_counter is not None:
                self._plugin_failure_counter.add(1, labels)
        except (AttributeError, TypeError):
            log.debug("graph_telemetry.metrics.record_failed plugin=%s", record.name)

    def start_run(
        self,
        run_id: str,
        repo: str,
        commit: str,
        plugin_count: int,
    ) -> GraphPluginSpan:
        """Start a telemetry span for an entire plugin run.

        Parameters
        ----------
        run_id
            Run identifier.
        repo
            Repository identifier.
        commit
            Commit SHA.
        plugin_count
            Number of plugins to execute.

        Returns
        -------
        GraphPluginSpan
            Span for the overall run.
        """
        span = GraphPluginSpan(
            plugin_name="__run__",
            run_id=run_id,
            start_time_ns=time.perf_counter_ns(),
            attributes={
                "repo": repo,
                "commit": commit,
                "plugin_count": plugin_count,
                "run_id": run_id,
            },
        )

        if self._tracer is not None:
            try:
                otel_span = self._tracer.start_span(
                    "graph.plugin_run",
                    attributes=span.attributes,
                )
                span.context_data["otel_span"] = otel_span
            except (ImportError, AttributeError):
                pass

        return span

    @staticmethod
    def finish_run(
        span: GraphPluginSpan,
        success_count: int,
        failure_count: int,
        skip_count: int,
    ) -> None:
        """Finish a run-level telemetry span.

        Parameters
        ----------
        span
            Span started with start_run.
        success_count
            Number of successful executions.
        failure_count
            Number of failed executions.
        skip_count
            Number of skipped executions.
        """
        duration_ns = time.perf_counter_ns() - span.start_time_ns
        duration_ms = duration_ns / 1_000_000

        log.info(
            "graph_telemetry.run.finish run_id=%s success=%d failed=%d skipped=%d duration_ms=%.2f",
            span.run_id,
            success_count,
            failure_count,
            skip_count,
            duration_ms,
        )

        otel_span = span.context_data.get("otel_span")
        if otel_span is not None:
            try:
                otel_span.set_attribute("success_count", success_count)
                otel_span.set_attribute("failure_count", failure_count)
                otel_span.set_attribute("skip_count", skip_count)
                otel_span.set_attribute("duration_ms", duration_ms)
                if StatusCode is not None:
                    status = StatusCode.OK if failure_count == 0 else StatusCode.ERROR
                    otel_span.set_status(status)
                otel_span.end()
            except AttributeError:
                pass


def get_graph_telemetry() -> GraphRuntimeTelemetry:
    """Return the default graph telemetry instance.

    Returns
    -------
    GraphRuntimeTelemetry
        Singleton telemetry instance.
    """
    return _telemetry_singleton()


@lru_cache(maxsize=1)
def _telemetry_singleton() -> GraphRuntimeTelemetry:
    return GraphRuntimeTelemetry()


__all__ = [
    "GraphPluginSpan",
    "GraphRuntimeTelemetry",
    "get_graph_telemetry",
]
