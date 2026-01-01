"""Shared utilities for serving HTTP route handlers."""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from anyio import EndOfStream, to_thread
from fastapi import BackgroundTasks

from codeintel.observability.operation_scope import observe_operation
from codeintel.observability.runtime import get_observability
from codeintel.observability.semconv import http_span_attributes
from codeintel.observability.semconv_keys import CODEINTEL_CORRELATION_ID, HTTP_ROUTE
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.metrics import QueryMetrics, log_query_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastapi import Request

    from codeintel.serving.operations.cancellation import CancelToken


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


class _UnsetResult:
    """Represent an unset threadpool result sentinel."""

    __slots__ = ()


_UNSET_RESULT = _UnsetResult()


@dataclass(frozen=True, slots=True)
class ThreadpoolMetricsContext[T]:
    """Context inputs for threadpool execution with metrics."""

    background: BackgroundTasks
    request: Request
    success_metrics: Callable[[T, float, str], QueryMetrics]
    error_metrics: Callable[[float, str], QueryMetrics]
    timeout_s: float | None = None
    cancel_token: CancelToken | None = None


async def run_in_threadpool_with_metrics[T](
    context: ThreadpoolMetricsContext[T],
    fn: Callable[..., T],
    *args: object,
    emit_metrics: bool = True,
    **kwargs: object,
) -> T:
    """Run a blocking operation in a threadpool and schedule query metrics.

    Parameters
    ----------
    context
        Execution context for metrics, timeouts, and cancellation.
    fn
        Blocking callable to execute in a threadpool.
    *args
        Positional arguments forwarded to `fn`.
    emit_metrics
        Whether to schedule metrics logging for the operation.
    **kwargs
        Keyword arguments forwarded to `fn`.

    Returns
    -------
    T
        The return value of `fn`.

    Raises
    ------
    TypeError
        If the threadpool operation did not produce a result.
    """
    correlation_id = get_correlation_id(context.request)
    route_label = _route_label(context.request)
    policy = get_observability().policy
    http_attrs = http_span_attributes(
        method=context.request.method,
        route=route_label,
        policy=policy,
    )
    normalized_route = http_attrs.get(HTTP_ROUTE)
    operation = (
        normalized_route if isinstance(normalized_route, str) and normalized_route else route_label
    )
    start = time.perf_counter()
    result: T | _UnsetResult = _UNSET_RESULT
    try:
        async with watch_request_disconnect(context.request, context.cancel_token):
            if context.cancel_token is not None:
                context.cancel_token.raise_if_cancelled()
            with observe_operation(
                component="http",
                operation=operation,
                attributes={
                    **http_attrs,
                    CODEINTEL_CORRELATION_ID: correlation_id,
                },
            ):
                if context.timeout_s is None:
                    result = await to_thread.run_sync(
                        lambda: fn(*args, **kwargs),
                        abandon_on_cancel=True,
                    )
                else:
                    with anyio.fail_after(context.timeout_s):
                        result = await to_thread.run_sync(
                            lambda: fn(*args, **kwargs),
                            abandon_on_cancel=True,
                        )
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        exc_type, _, _ = sys.exc_info()
    if emit_metrics:
        if exc_type is None and not isinstance(result, _UnsetResult):
            schedule_query_metrics(
                context.background,
                context.success_metrics(result, duration_ms, correlation_id),
            )
        else:
            schedule_query_metrics(
                context.background,
                context.error_metrics(duration_ms, correlation_id),
            )
    if isinstance(result, _UnsetResult):
        msg = "Threadpool execution returned no result"
        raise TypeError(msg)
    return result


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


_DISCONNECT_POLL_INTERVAL_S = 0.25


async def _watch_disconnect(request: Request, cancel_token: CancelToken) -> None:
    while True:
        try:
            with anyio.move_on_after(_DISCONNECT_POLL_INTERVAL_S) as scope:
                disconnected = await request.is_disconnected()
            if scope.cancel_called:
                disconnected = False
        except (EndOfStream, RuntimeError):
            cancel_token.cancel(reason="client disconnected")
            return
        if disconnected:
            cancel_token.cancel(reason="client disconnected")
            return
        await anyio.sleep(_DISCONNECT_POLL_INTERVAL_S)


@asynccontextmanager
async def watch_request_disconnect(
    request: Request, cancel_token: CancelToken | None
) -> AsyncIterator[None]:
    """Cancel the token when the request disconnects."""
    if cancel_token is None:
        yield
        return
    client = request.scope.get("client")
    if isinstance(client, tuple) and client and client[0] == "testclient":
        yield
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        yield
        return
    task = loop.create_task(_watch_disconnect(request, cancel_token))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


__all__ = [
    "ThreadpoolMetricsContext",
    "run_in_threadpool_with_metrics",
    "schedule_query_metrics",
    "watch_request_disconnect",
]
