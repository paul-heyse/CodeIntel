"""Tracing middleware for plugin execution.

This module provides middleware that creates trace spans for
plugin execution, enabling distributed tracing integration.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.analytics.core.protocol import (
        AnalyticsPluginProtocol,
        PluginResult,
    )


@dataclass(frozen=True)
class SpanContext:
    """Context for a trace span.

    Attributes
    ----------
    trace_id
        Unique trace identifier.
    span_id
        Unique span identifier.
    parent_span_id
        Parent span identifier, if any.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None


@dataclass
class Span:
    """A trace span for plugin execution.

    Attributes
    ----------
    name
        Span name.
    context
        Span context.
    start_time
        When the span started.
    end_time
        When the span ended.
    status
        Span status ("ok" or "error").
    attributes
        Span attributes.
    """

    name: str
    context: SpanContext
    start_time: float
    end_time: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished.

        Parameters
        ----------
        status
            Final status.
        """
        self.end_time = time.perf_counter()
        self.status = status

    @property
    def duration_ms(self) -> float:
        """Return span duration in milliseconds.

        Returns
        -------
        float
            Duration, or 0 if not finished.
        """
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class SpanExporter:
    """Base class for span exporters.

    Override export() to send spans to your tracing backend.
    """

    def export(self, span: Span) -> None:
        """Export a completed span.

        Parameters
        ----------
        span
            Span to export.
        """


class InMemoryExporter(SpanExporter):
    """Exporter that stores spans in memory (for testing)."""

    def __init__(self) -> None:
        """Initialize the exporter."""
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        """Store the span in memory.

        Parameters
        ----------
        span
            Span to store.
        """
        self.spans.append(span)

    def clear(self) -> None:
        """Clear stored spans."""
        self.spans.clear()


def _generate_id() -> str:
    """Generate a unique ID for traces/spans.

    Returns
    -------
    str
        A 16-character hex string.
    """
    return secrets.token_hex(8)


@dataclass
class TracingMiddleware:
    """Middleware that creates trace spans for plugin execution.

    Creates spans with:
    - Plugin name as span name
    - Execution metadata as attributes
    - Error status on failure

    Attributes
    ----------
    exporter
        Span exporter to use.
    trace_id
        Current trace ID (set per-run).
    """

    exporter: SpanExporter = field(default_factory=SpanExporter)
    trace_id: str | None = None

    _active_spans: dict[str, Span] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        """Return middleware name."""
        return "tracing"

    def _get_trace_id(self, ctx: PluginExecutionContext) -> str:
        """Get or create trace ID for a run.

        Returns
        -------
        str
            The trace ID for correlation.
        """
        if self.trace_id is not None:
            return self.trace_id
        # Use run_id as trace_id for correlation, or generate a default
        return ctx.run_id or "no-run-id"

    def before_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
    ) -> None:
        """Create and start a span for the plugin.

        Parameters
        ----------
        ctx
            Execution context.
        plugin
            Plugin about to execute.
        """
        plugin_name = plugin.metadata.name
        trace_id = self._get_trace_id(ctx)

        context = SpanContext(
            trace_id=trace_id,
            span_id=_generate_id(),
        )

        span = Span(
            name=f"plugin.{plugin_name}",
            context=context,
            start_time=time.perf_counter(),
            attributes={
                "plugin.name": plugin_name,
                "plugin.version": plugin.metadata.version,
                "plugin.stage": plugin.metadata.stage,
                "run.id": ctx.run_id,
                "repo": ctx.repo,
                "commit": ctx.commit,
            },
        )

        self._active_spans[plugin_name] = span

    def after_execute(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        result: PluginResult,
    ) -> PluginResult:
        """Finish the span with result data.

        Parameters
        ----------
        ctx
            Execution context (required by interface).
        plugin
            Plugin that executed.
        result
            Execution result.

        Returns
        -------
        PluginResult
            Unchanged result.
        """
        plugin_name = plugin.metadata.name
        span = self._active_spans.pop(plugin_name, None)

        if span is not None:
            span.attributes["result.success"] = result.success
            span.attributes.setdefault("repo", ctx.repo)
            span.attributes.setdefault("commit", ctx.commit)
            if ctx.run_id is not None:
                span.attributes.setdefault("run.id", ctx.run_id)

            if result.row_counts:
                span.attributes["result.row_counts"] = dict(result.row_counts)
                span.attributes["result.total_rows"] = sum(result.row_counts.values())

            if result.error:
                span.attributes["error"] = str(result.error)

            status = "ok" if result.success else "error"
            span.finish(status)
            self.exporter.export(span)

        return result

    def on_error(
        self,
        ctx: PluginExecutionContext,
        plugin: AnalyticsPluginProtocol,
        error: Exception,
    ) -> Exception | None:
        """Finish the span with error status.

        Parameters
        ----------
        ctx
            Execution context (required by interface).
        plugin
            Plugin that raised.
        error
            The exception raised.

        Returns
        -------
        Exception
            The error unchanged.
        """
        plugin_name = plugin.metadata.name
        span = self._active_spans.pop(plugin_name, None)

        if span is not None:
            span.attributes["error.type"] = type(error).__name__
            span.attributes["error.message"] = str(error)
            span.attributes.setdefault("repo", ctx.repo)
            span.attributes.setdefault("commit", ctx.commit)
            if ctx.run_id is not None:
                span.attributes.setdefault("run.id", ctx.run_id)
            span.finish("error")
            self.exporter.export(span)

        return error


__all__ = [
    "InMemoryExporter",
    "Span",
    "SpanContext",
    "SpanExporter",
    "TracingMiddleware",
]
