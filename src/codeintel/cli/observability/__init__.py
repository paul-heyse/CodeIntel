"""Observability infrastructure for CLI operations.

This package provides:

- ``TelemetryProvider``: OpenTelemetry integration
- ``ObservabilityMiddleware``: Tracing and metrics middleware
- ``OperationMetrics``: Metrics collection
- Structured logging configuration
"""

from __future__ import annotations

# Re-export from observability module
from codeintel.cli.observability._observability import (
    ObservabilityConfig,
    ObservabilityMiddleware,
    StructuredLogFormatter,
    configure_structured_logging,
    get_observability_middleware,
)

# Re-export from telemetry module
from codeintel.cli.observability._telemetry import (
    OperationMetrics,
    TelemetryConfig,
    TelemetryProvider,
    TracingMiddleware,
    get_operation_metrics,
    get_telemetry_provider,
)

__all__ = [
    "ObservabilityConfig",
    "ObservabilityMiddleware",
    "OperationMetrics",
    "StructuredLogFormatter",
    "TelemetryConfig",
    "TelemetryProvider",
    "TracingMiddleware",
    "configure_structured_logging",
    "get_observability_middleware",
    "get_operation_metrics",
    "get_telemetry_provider",
]
