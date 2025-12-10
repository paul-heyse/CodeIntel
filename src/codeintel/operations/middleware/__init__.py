"""Middleware for cross-cutting concerns in operation execution.

Middleware wraps operation execution to add logging, metrics, validation,
and error handling consistently across all adapters.
"""

from __future__ import annotations

from codeintel.operations.middleware.base import BaseMiddleware, OperationMiddleware
from codeintel.operations.middleware.errors import ErrorHandlingMiddleware
from codeintel.operations.middleware.logging import LoggingMiddleware
from codeintel.operations.middleware.telemetry import TelemetryMiddleware
from codeintel.operations.middleware.validation import ValidationError, ValidationMiddleware

__all__ = [
    "BaseMiddleware",
    "ErrorHandlingMiddleware",
    "LoggingMiddleware",
    "OperationMiddleware",
    "TelemetryMiddleware",
    "ValidationError",
    "ValidationMiddleware",
]
