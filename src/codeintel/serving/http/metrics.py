"""Background metrics logging for serving HTTP routes.

This module provides data structures and functions for capturing and logging
query metrics in a non-blocking manner using FastAPI's background tasks.
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
        HTTP endpoint path (e.g., "/semantic/query").
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
    """

    endpoint: str
    view_id: str | None
    query: str | None
    row_count: int
    truncated: bool
    duration_ms: float
    correlation_id: str
    engine: str | None = None


def log_query_metrics(metrics: QueryMetrics) -> None:
    """Log query metrics as structured log entry.

    Designed to run as a FastAPI background task to avoid blocking
    the response. The log is emitted at INFO level with extra fields
    suitable for structured log aggregation.

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
        },
    )


__all__ = ["QueryMetrics", "log_query_metrics"]
