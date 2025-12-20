"""Shared utilities for serving HTTP route handlers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeVar

from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.metrics import QueryMetrics, log_query_metrics

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request


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


T = TypeVar("T")


async def run_in_threadpool_with_metrics(
    background: BackgroundTasks,
    request: Request,
    fn: Callable[..., T],
    success_metrics: Callable[[T, float, str], QueryMetrics],
    error_metrics: Callable[[float, str], QueryMetrics],
    *args: object,
    **kwargs: object,
) -> T:
    """Run a blocking operation in a threadpool and schedule query metrics.

    Parameters
    ----------
    background
        Background task queue.
    request
        Incoming HTTP request used to extract correlation IDs.
    fn
        Blocking callable to execute in a threadpool.
    success_metrics
        Callback that builds a metrics payload on success; receives `(result, duration_ms, correlation_id)`.
    error_metrics
        Callback that builds a metrics payload on error; receives `(duration_ms, correlation_id)`.
    *args
        Positional arguments forwarded to `fn`.
    **kwargs
        Keyword arguments forwarded to `fn`.

    Returns
    -------
    T
        The return value of `fn`.
    """
    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        result = await run_in_threadpool(fn, *args, **kwargs)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(background, error_metrics(duration_ms, correlation_id))
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    schedule_query_metrics(background, success_metrics(result, duration_ms, correlation_id))
    return result


__all__ = ["run_in_threadpool_with_metrics", "schedule_query_metrics"]
