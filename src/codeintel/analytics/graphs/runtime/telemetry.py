"""Telemetry hooks for graph runtime execution."""

from __future__ import annotations

from typing import Protocol, cast

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from codeintel.analytics.graphs.plugins import GraphMetricExecutionContext, GraphMetricPlugin
from codeintel.analytics.graphs.runtime.manifest import hash_json
from codeintel.analytics.graphs.runtime.model import GraphPluginRunRecord
from codeintel.config.steps_graphs import GraphRunScope


class GraphRuntimeTelemetry(Protocol):
    """Protocol describing telemetry hooks for graph runtime events."""

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> object:
        """
        Start telemetry for a plugin run and return an opaque span object.

        Returns
        -------
        object
            Span or token representing the started telemetry scope.
        """

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
        """Complete telemetry span for the plugin run."""

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        """Record metrics for a completed plugin run."""


class NoOpGraphRuntimeTelemetry:
    """Telemetry implementation that emits nothing."""

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> None:
        """Return a no-op span placeholder."""
        _ = self
        _ = plugin
        _ = run_id
        _ = ctx

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
        """No-op span finisher."""
        _ = self
        _ = span
        _ = record

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        """No-op metrics recorder."""
        _ = self
        _ = record
        _ = scope


class OtelGraphRuntimeTelemetry:
    """OpenTelemetry-backed telemetry for graph plugins."""

    def __init__(self) -> None:
        self.tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        self.duration_ms = meter.create_histogram(
            "graph_plugin_duration_ms",
            unit="ms",
            description="Duration of graph metric plugins",
        )
        self.status_counter = meter.create_counter(
            "graph_plugin_status_total",
            description="Count of graph metric plugin executions by status",
        )
        self.retry_counter = meter.create_counter(
            "graph_plugin_retries_total",
            description="Count of retry attempts for graph metric plugins",
        )
        self.skip_counter = meter.create_counter(
            "graph_plugin_skipped_total",
            description="Count of skipped graph metric plugins by reason",
        )

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> object:
        """
        Start an OTEL span for the plugin execution.

        Returns
        -------
        object
            OTEL span representing the plugin execution.
        """
        scope_payload = {
            "paths": ctx.scope.paths,
            "modules": ctx.scope.modules,
            "time_window": (
                (
                    ctx.scope.time_window[0].isoformat(),
                    ctx.scope.time_window[1].isoformat(),
                )
                if ctx.scope.time_window is not None
                else None
            ),
        }
        attributes = {
            "graph.plugin": plugin.name,
            "graph.stage": plugin.stage,
            "graph.run_id": run_id,
            "graph.scope.paths": len(ctx.scope.paths),
            "graph.scope.modules": len(ctx.scope.modules),
            "graph.scope.time_window": ctx.scope.time_window is not None,
            "graph.repo": ctx.repo,
            "graph.commit": ctx.commit,
            "graph.scope_hash": hash_json(scope_payload) if scope_payload else "none",
            "graph.options_hash": hash_json(ctx.options) if ctx.options is not None else "none",
        }
        return self.tracer.start_span("graph.plugin", attributes=attributes)

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
        """Finish the OTEL span with status and error metadata."""
        _ = self
        span_obj = cast("trace.Span", span)
        span_obj.set_attribute("graph.status", record.status)
        span_obj.set_attribute("graph.attempts", record.attempts)
        span_obj.set_attribute("graph.severity", record.severity)
        span_obj.set_attribute("graph.partial", record.partial)
        span_obj.set_attribute("graph.requires_isolation", record.requires_isolation)
        span_obj.set_attribute("graph.policy_fail_fast", record.policy_fail_fast)
        span_obj.set_attribute("graph.input_hash", record.input_hash or "none")
        span_obj.set_attribute("graph.version_hash", record.version_hash or "none")
        span_obj.set_attribute("graph.timeout_ms", record.timeout_ms or 0)
        span_obj.set_attribute(
            "graph.options_hash",
            hash_json(record.options) if record.options is not None else "none",
        )
        if record.isolation_kind is not None:
            span_obj.set_attribute("graph.isolation_kind", record.isolation_kind)
        if record.error is not None:
            span_obj.record_exception(Exception(record.error))
            span_obj.set_status(Status(StatusCode.ERROR, record.error))
        else:
            span_obj.set_status(Status(StatusCode.OK))
        span_obj.end()

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        """Emit OTEL metrics for plugin execution duration and status."""
        scope_present = bool(scope.paths or scope.modules or scope.time_window)
        scope_payload = {
            "paths": scope.paths,
            "modules": scope.modules,
            "time_window": scope.time_window,
        }
        scope_hash = hash_json(scope_payload) if scope_present else "none"
        options_hash = hash_json(record.options) if record.options is not None else "none"
        attributes = {
            "plugin": record.name,
            "stage": record.stage,
            "severity": record.severity,
            "status": record.status,
            "requires_isolation": record.requires_isolation,
            "isolation_kind": record.isolation_kind or "none",
            "scope_paths": len(scope.paths),
            "scope_modules": len(scope.modules),
            "scope_time_window": scope.time_window is not None,
            "scope_present": scope_present,
            "scope_hash": scope_hash,
            "options_hash": options_hash,
            "policy_fail_fast": record.policy_fail_fast,
        }
        self.duration_ms.record(record.duration_ms, attributes=attributes)
        self.status_counter.add(1, attributes=attributes)
        if record.attempts > 1:
            self.retry_counter.add(record.attempts - 1, attributes=attributes)
        if record.status == "skipped":
            self.skip_counter.add(
                1,
                attributes={**attributes, "skip_reason": record.skipped_reason or "unspecified"},
            )


__all__ = [
    "GraphRuntimeTelemetry",
    "NoOpGraphRuntimeTelemetry",
    "OtelGraphRuntimeTelemetry",
]
