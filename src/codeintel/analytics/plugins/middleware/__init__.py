"""Plugin middleware for cross-cutting concerns.

This package provides middleware that wraps plugin execution to handle
cross-cutting concerns like logging, metrics, and tracing.

Middleware Chain
----------------
Middleware is applied in a chain pattern:
1. before_execute is called for each middleware (in order)
2. The plugin executes
3. after_execute is called for each middleware (in reverse order)

This allows middleware to:
- Prepare state before execution
- Transform or augment results
- Handle errors and cleanup

Example
-------
>>> from codeintel.analytics.plugins.middleware import LoggingMiddleware
>>> executor = PluginExecutor(middleware=[LoggingMiddleware()])
"""

from __future__ import annotations

from codeintel.analytics.plugins.middleware.logging import LoggingMiddleware
from codeintel.analytics.plugins.middleware.metrics import MetricsMiddleware
from codeintel.analytics.plugins.middleware.protocol import (
    MiddlewareChain,
    PluginMiddleware,
)
from codeintel.analytics.plugins.middleware.tracing import TracingMiddleware

__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareChain",
    "PluginMiddleware",
    "TracingMiddleware",
]
