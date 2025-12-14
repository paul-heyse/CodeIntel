"""Unified observability infrastructure.

This module provides core observability patterns for the codebase,
including metrics, tracing, and protocols.

Examples
--------
Using in-memory metrics:

>>> from codeintel.core.observability import InMemoryMetrics, timed_metric
>>>
>>> metrics = InMemoryMetrics()
>>> metrics.increment("requests_total", 1, method="GET")
>>> with timed_metric(metrics, "request_duration_seconds"):
...     process_request()

Using in-memory tracing:

>>> from codeintel.core.observability import InMemoryTracer, trace_operation
>>>
>>> tracer = InMemoryTracer()
>>> with trace_operation(tracer, "process") as span:
...     span.set_attribute("user_id", "123")
...     do_work()
"""

from codeintel.core.observability.metrics import (
    InMemoryMetrics,
    MetricValue,
    timed_metric,
)
from codeintel.core.observability.protocol import (
    MetricsProtocol,
    ObservabilityProtocol,
    SpanProtocol,
    TracingProtocol,
)
from codeintel.core.observability.tracing import (
    InMemoryTracer,
    Span,
    trace_operation,
)

__all__ = [
    "InMemoryMetrics",
    "InMemoryTracer",
    "MetricValue",
    "MetricsProtocol",
    "ObservabilityProtocol",
    "Span",
    "SpanProtocol",
    "TracingProtocol",
    "timed_metric",
    "trace_operation",
]
