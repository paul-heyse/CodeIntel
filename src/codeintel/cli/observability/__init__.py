"""Observability infrastructure for CLI operations.

This package provides:

- ``TelemetryProvider``: OpenTelemetry integration
- ``OperationMetrics``: Metrics collection
- Structured logging configuration
"""

from __future__ import annotations

from codeintel.cli.observability._observability import (
    ObservabilityConfig,
    StructuredLogFormatter,
    configure_structured_logging,
)
from codeintel.cli.observability._telemetry import (
    OperationMetrics,
    TelemetryConfig,
    TelemetryProvider,
    get_operation_metrics,
    get_telemetry_provider,
)

__all__ = [
    "ObservabilityConfig",
    "OperationMetrics",
    "StructuredLogFormatter",
    "TelemetryConfig",
    "TelemetryProvider",
    "configure_structured_logging",
    "get_operation_metrics",
    "get_telemetry_provider",
]
