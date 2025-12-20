"""Structured metrics logging for serving operations.

This module provides data structures and helpers for capturing and logging
query metrics in a transport-agnostic way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger("codeintel.serving.metrics")


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Captured metrics for a query execution.

    Parameters
    ----------
    endpoint
        Endpoint identifier (HTTP path or MCP tool name).
    view_id
        Semantic view identifier if applicable.
    query
        Search query text if applicable.
    row_count
        Number of rows returned.
    truncated
        Whether results were truncated by limit.
    duration_ms
        Query execution time in milliseconds.
    correlation_id
        Request correlation identifier.
    engine
        Result extraction engine used (polars/pandas).
    query_hash
        Stable fingerprint of query inputs when available.
    schema_hash
        Stable fingerprint of resolved schema when available.
    """

    endpoint: str
    view_id: str | None
    query: str | None
    row_count: int
    truncated: bool
    duration_ms: float
    correlation_id: str
    engine: str | None = None
    query_hash: str | None = None
    schema_hash: str | None = None


def log_query_metrics(metrics: QueryMetrics) -> None:
    """Log query metrics as structured log entry.

    Parameters
    ----------
    metrics
        Captured query metrics to log.
    """
    LOG.info(
        "query_executed",
        extra={
            "endpoint": metrics.endpoint,
            "view_id": metrics.view_id,
            "query": metrics.query,
            "row_count": metrics.row_count,
            "truncated": metrics.truncated,
            "duration_ms": round(metrics.duration_ms, 3),
            "correlation_id": metrics.correlation_id,
            "engine": metrics.engine,
            "query_hash": metrics.query_hash,
            "schema_hash": metrics.schema_hash,
        },
    )


__all__ = ["QueryMetrics", "log_query_metrics"]
