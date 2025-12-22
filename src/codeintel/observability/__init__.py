"""Shared observability utilities for CLI, HTTP, MCP, and storage."""

from __future__ import annotations

from codeintel.observability.context import (
    correlation_context,
    get_correlation_id,
    set_correlation_id,
)
from codeintel.observability.duckdb_tracing import maybe_instrument_duckdb_connection
from codeintel.observability.mcp import McpOpenTelemetryMiddleware
from codeintel.observability.operations import (
    observe_operation,
    record_operation_metrics,
    record_query_metrics,
)
from codeintel.observability.otel import (
    ObservabilityConfig,
    bootstrap_observability,
    get_observability,
    shutdown_observability,
)
from codeintel.observability.sql_redaction import RedactedSQL, SQLStatementMode, redact_sql

__all__ = [
    "McpOpenTelemetryMiddleware",
    "ObservabilityConfig",
    "RedactedSQL",
    "SQLStatementMode",
    "bootstrap_observability",
    "correlation_context",
    "get_correlation_id",
    "get_observability",
    "maybe_instrument_duckdb_connection",
    "observe_operation",
    "record_operation_metrics",
    "record_query_metrics",
    "redact_sql",
    "set_correlation_id",
    "shutdown_observability",
]
