"""Shared observability utilities for CLI, HTTP, MCP, and storage."""

from __future__ import annotations

from codeintel.observability.db_tracing import RedactedSQL, SQLStatementMode, redact_sql
from codeintel.observability.duckdb_tracing import maybe_instrument_duckdb_connection
from codeintel.observability.lifecycle import ObservabilityLifecycle
from codeintel.observability.mcp import McpOpenTelemetryMiddleware
from codeintel.observability.operation_scope import (
    observe_operation,
    record_operation_metrics,
    record_query_metrics,
)
from codeintel.observability.runtime import (
    ConfigResolver,
    ObservabilityConfig,
    ResolvedObservabilityConfig,
    bootstrap_observability,
    get_observability,
    resolve_observability_config,
    shutdown_observability,
)
from codeintel.observability.telemetry_context import (
    current_telemetry_context,
    telemetry_context,
)

__all__ = [
    "ConfigResolver",
    "McpOpenTelemetryMiddleware",
    "ObservabilityConfig",
    "ObservabilityLifecycle",
    "RedactedSQL",
    "ResolvedObservabilityConfig",
    "SQLStatementMode",
    "bootstrap_observability",
    "current_telemetry_context",
    "get_observability",
    "maybe_instrument_duckdb_connection",
    "observe_operation",
    "record_operation_metrics",
    "record_query_metrics",
    "redact_sql",
    "resolve_observability_config",
    "shutdown_observability",
    "telemetry_context",
]
