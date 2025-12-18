"""Shared utilities for serving HTTP route handlers."""

from __future__ import annotations

from fastapi import BackgroundTasks

from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics


def schedule_query_metrics(background: BackgroundTasks, metrics: QueryMetrics) -> None:
    """Schedule query metrics logging on FastAPI background tasks.

    Parameters
    ----------
    background
        Background task queue.
    metrics
        Captured query metrics.
    """
    background.add_task(log_query_metrics, metrics)


__all__ = ["schedule_query_metrics"]
