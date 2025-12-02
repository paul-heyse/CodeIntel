"""Middleware for ingestion plugin execution.

This package provides middleware components for cross-cutting concerns
in plugin execution, such as logging, metrics, and tracing.

Middleware follows the before/after/on_error pattern:
- `before_execute`: Called before plugin execution
- `after_execute`: Called after successful execution
- `on_error`: Called when an error occurs

Example
-------
>>> from codeintel.ingestion.core.middleware import LoggingMiddleware, MetricsMiddleware
>>> executor = PluginExecutor(
...     middleware=[LoggingMiddleware(), MetricsMiddleware()],
... )
"""

from __future__ import annotations

from codeintel.ingestion.core.middleware.base import (
    IngestMiddleware,
    MiddlewareChain,
)
from codeintel.ingestion.core.middleware.logging import LoggingMiddleware
from codeintel.ingestion.core.middleware.metrics import MetricsMiddleware
from codeintel.ingestion.core.middleware.tracing import TracingMiddleware

__all__ = [
    "IngestMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareChain",
    "TracingMiddleware",
]
