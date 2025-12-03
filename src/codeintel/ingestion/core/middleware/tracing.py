"""Tracing middleware for ingestion plugin execution.

This module provides distributed tracing support for plugin execution,
compatible with OpenTelemetry spans.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from codeintel.ingestion.core.base import BaseIngestPlugin
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.plugins.protocol import IngestPluginResult

log = logging.getLogger(__name__)


class SpanContext(Protocol):
    """Protocol for OpenTelemetry-style span context."""

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute on the span.

        Parameters
        ----------
        key
            Attribute key.
        value
            Attribute value.
        """
        ...

    def set_status(self, status: object, description: str | None = None) -> None:
        """Set the span status.

        Parameters
        ----------
        status
            Status value.
        description
            Optional description.
        """
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span.

        Parameters
        ----------
        exception
            The exception to record.
        """
        ...

    def end(self) -> None:
        """End the span."""
        ...


class Tracer(Protocol):
    """Protocol for OpenTelemetry-style tracer."""

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> SpanContext:
        """Start a new span.

        Parameters
        ----------
        name
            Span name.
        attributes
            Initial span attributes.

        Returns
        -------
        SpanContext
            The started span.
        """
        ...


@dataclass
class InMemorySpan:
    """In-memory span for testing.

    Collect span data in memory for inspection.
    """

    name: str
    attributes: dict[str, object] = field(default_factory=dict)
    status: str = "unset"
    status_description: str | None = None
    exception: Exception | None = None
    ended: bool = False

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute.

        Parameters
        ----------
        key
            Attribute key.
        value
            Attribute value.
        """
        self.attributes[key] = value

    def set_status(self, status: object, description: str | None = None) -> None:
        """Set the status.

        Parameters
        ----------
        status
            Status value (converted to string).
        description
            Optional description.
        """
        self.status = str(status)
        self.status_description = description

    def record_exception(self, exception: Exception) -> None:
        """Record an exception.

        Parameters
        ----------
        exception
            The exception to record.
        """
        self.exception = exception

    def end(self) -> None:
        """End the span."""
        self.ended = True


@dataclass
class InMemoryTracer:
    """In-memory tracer for testing.

    Collect spans in memory for inspection.
    """

    spans: list[InMemorySpan] = field(default_factory=list)

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> InMemorySpan:
        """Start a new span.

        Parameters
        ----------
        name
            Span name.
        attributes
            Initial attributes.

        Returns
        -------
        InMemorySpan
            The created span.
        """
        span = InMemorySpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(span)
        return span

    def clear(self) -> None:
        """Clear all spans."""
        self.spans.clear()


@dataclass
class TracingMiddleware:
    """Middleware that creates spans for plugin execution.

    Create distributed tracing spans for plugin execution.
    Supports both OpenTelemetry tracers and in-memory tracing.

    Attributes
    ----------
    tracer
        Optional OpenTelemetry-compatible tracer.
    in_memory
        Optional in-memory tracer for testing.
    span_name_prefix
        Prefix for span names.
    """

    tracer: Tracer | None = None
    in_memory: InMemoryTracer | None = None
    span_name_prefix: str = "ingest.plugin"
    _active_spans: dict[str, SpanContext | InMemorySpan] = field(default_factory=dict, repr=False)

    def _get_effective_tracer(self) -> Tracer | InMemoryTracer | None:
        """Get the tracer to use.

        Returns
        -------
        Tracer | InMemoryTracer | None
            The tracer, or None if no tracer is configured.
        """
        return self.tracer or self.in_memory

    def before_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
    ) -> None:
        """Start a span for plugin execution.

        Parameters
        ----------
        plugin
            The plugin about to execute.
        ctx
            Execution context.
        """
        tracer = self._get_effective_tracer()
        if tracer is None:
            return

        plugin_name = plugin.metadata.name
        span_name = f"{self.span_name_prefix}.{plugin_name}"

        attributes: dict[str, Any] = {
            "plugin.name": plugin_name,
            "plugin.stage": plugin.metadata.stage,
            "repo": ctx.repo,
            "commit": ctx.commit,
        }

        ctx.start_plugin_timer(plugin_name)
        span = tracer.start_span(span_name, attributes)
        self._active_spans[plugin_name] = span

    def after_execute(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        result: IngestPluginResult,
    ) -> None:
        """End the span with result information.

        Parameters
        ----------
        plugin
            The plugin that executed.
        ctx
            Execution context (unused, required by protocol).
        result
            Execution result.
        """
        plugin_name = plugin.metadata.name
        span = self._active_spans.pop(plugin_name, None)
        if span is None:
            return

        duration_s = ctx.finish_plugin_timer(plugin_name)
        span.set_attribute("result.duration_ms", round(duration_s * 1000, 2))
        # Add result attributes
        span.set_attribute("result.success", result.success)
        span.set_attribute("result.skipped", result.skipped)

        if result.skipped:
            span.set_attribute("result.skip_reason", result.skip_reason or "unknown")
            span.set_status("ok", "skipped")
        elif result.success:
            if result.row_counts:
                total_rows = sum(result.row_counts.values())
                span.set_attribute("result.total_rows", total_rows)
            span.set_status("ok")
        else:
            span.set_attribute("result.error", result.error or "unknown")
            span.set_attribute("result.error_kind", result.error_kind or "unknown")
            span.set_status("error", result.error)

        span.end()

    def on_error(
        self,
        plugin: BaseIngestPlugin,
        ctx: IngestExecutionContext,
        error: Exception,
    ) -> None:
        """End the span with error information.

        Parameters
        ----------
        plugin
            The plugin that failed.
        ctx
            Execution context (unused, required by protocol).
        error
            The exception that was raised.
        """
        plugin_name = plugin.metadata.name
        span = self._active_spans.pop(plugin_name, None)
        if span is None:
            return

        duration_s = ctx.finish_plugin_timer(plugin_name)
        span.set_attribute("result.duration_ms", round(duration_s * 1000, 2))
        span.record_exception(error)
        span.set_status("error", str(error))
        span.end()


__all__ = [
    "InMemorySpan",
    "InMemoryTracer",
    "SpanContext",
    "Tracer",
    "TracingMiddleware",
]
