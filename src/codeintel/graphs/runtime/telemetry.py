"""Telemetry utilities for graph plugin execution.

This module provides graph-specific telemetry that extends the base
RuntimeTelemetry from core with graph-specific span tracking and metrics.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from codeintel.core.execution.telemetry import (
    OTEL_AVAILABLE,
    PluginSpan,
    RuntimeTelemetry,
    TelemetryConfig,
)
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from opentelemetry.trace import StatusCode as _StatusCodeType

    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.core.plugins.types.result import PluginExecutionRecord
    from codeintel.graphs.core.context import GraphPluginExecutionContext
    from codeintel.graphs.core.protocol import GraphPluginProtocol

# Runtime-safe optional import for StatusCode
# Use _StatusCode to access; will be None if OTel not available
_StatusCode: type[_StatusCodeType] | None = None
if OTEL_AVAILABLE:
    from opentelemetry.trace import StatusCode as _StatusCode

log = logging.getLogger(__name__)

# Alias for backwards compatibility
GraphPluginSpan = PluginSpan


class GraphTelemetryConfig(TelemetryConfig):
    """Configuration for graph telemetry."""

    service_name: str = "codeintel.graphs"


class GraphRuntimeTelemetry(RuntimeTelemetry):
    """Telemetry manager for graph plugin execution.

    Extend the base RuntimeTelemetry with graph-specific methods for
    tracking plugin execution with context information, run-level spans,
    and scope-aware metrics.

    Examples
    --------
    >>> telemetry = get_graph_telemetry()
    >>> span = telemetry.start_plugin(plugin, run_id, ctx)
    >>> try:
    ...     result = plugin.execute(ctx)
    ...     telemetry.finish_plugin(span, record)
    ... except Exception:
    ...     telemetry.finish_plugin(span, record)
    """

    def __init__(
        self,
        *,
        service_name: str = "codeintel.graphs",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize graph telemetry.

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

    def start_plugin(
        self,
        plugin: GraphPluginProtocol,
        run_id: str,
        ctx: GraphPluginExecutionContext,
    ) -> PluginSpan:
        """Start a telemetry span for plugin execution with graph context.

        Parameters
        ----------
        plugin
            Plugin being executed.
        run_id
            Run identifier.
        ctx
            Execution context with repo and commit info.

        Returns
        -------
        PluginSpan
            Span object for tracking execution.
        """
        return self.start_span(
            plugin.metadata.name,
            run_id,
            attributes={
                "plugin.name": plugin.metadata.name,
                "plugin.kind": plugin.metadata.kind,
                "plugin.stage": plugin.metadata.stage,
                "repo": ctx.repo,
                "commit": ctx.commit,
                "run_id": run_id,
            },
        )

    @staticmethod
    def finish_plugin(
        span: PluginSpan,
        record: PluginExecutionRecord,
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
                span_any = cast("Any", otel_span)
                span_any.set_attribute("status", record.status)
                span_any.set_attribute("duration_ms", duration_ms)
                span_any.set_attribute("attempts", record.attempts)
                if record.error and _StatusCode is not None:
                    span_any.set_attribute("error", record.error)
                    span_any.set_status(_StatusCode.ERROR, record.error)
                elif _StatusCode is not None:
                    span_any.set_status(_StatusCode.OK)
                span_any.end()
            except AttributeError:
                pass

    def record_metrics(
        self,
        record: PluginExecutionRecord,
        scope: GraphRunScope | None,
    ) -> None:
        """Record execution metrics with optional scope information.

        Parameters
        ----------
        record
            Execution record with results.
        scope
            Optional execution scope for additional labels.
        """
        if not self.metrics_enabled:
            return

        labels: dict[str, str | int | float | bool] = {
            "plugin_name": record.plugin_name,
            "status": record.status,
        }
        if scope is not None:
            if scope.paths:
                labels["scope_path_count"] = len(scope.paths)
            if scope.modules:
                labels["scope_module_count"] = len(scope.modules)

        try:
            if self._otel_duration_histogram is not None:
                self._otel_duration_histogram.record(record.duration_ms, labels)

            if self._prom_duration_histogram is not None:
                self._prom_duration_histogram.labels(
                    plugin_name=record.plugin_name,
                    status=record.status,
                ).observe(record.duration_ms / 1000)  # Convert to seconds

            if self._prom_executions_counter is not None:
                self._prom_executions_counter.labels(
                    plugin_name=record.plugin_name,
                    status=record.status,
                ).inc()
        except (AttributeError, TypeError):
            log.debug("graph_telemetry.metrics.record_failed plugin=%s", record.plugin_name)

    def start_run(
        self,
        run_id: str,
        repo: str,
        commit: str,
        plugin_count: int,
    ) -> PluginSpan:
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
        PluginSpan
            Span for the overall run.
        """
        return self.start_span(
            "__run__",
            run_id,
            attributes={
                "repo": repo,
                "commit": commit,
                "plugin_count": plugin_count,
            },
        )

    @staticmethod
    def finish_run(
        span: PluginSpan,
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
                if _StatusCode is not None:
                    status = _StatusCode.OK if failure_count == 0 else _StatusCode.ERROR
                    otel_span.set_status(status)
                otel_span.end()
            except AttributeError:
                pass


class _GraphTelemetryHolder(SingletonHolder[GraphRuntimeTelemetry]):
    """Singleton holder for GraphRuntimeTelemetry."""


def get_graph_telemetry() -> GraphRuntimeTelemetry:
    """Return the default graph telemetry singleton.

    Returns
    -------
    GraphRuntimeTelemetry
        Shared telemetry instance.
    """
    return _GraphTelemetryHolder.get(GraphRuntimeTelemetry)


def reset_graph_telemetry() -> None:
    """Reset the telemetry singleton for testing."""
    _GraphTelemetryHolder.reset()


__all__ = [
    "GraphPluginSpan",
    "GraphRuntimeTelemetry",
    "GraphTelemetryConfig",
    "get_graph_telemetry",
    "reset_graph_telemetry",
]
