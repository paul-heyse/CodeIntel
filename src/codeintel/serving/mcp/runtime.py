"""MCP runtime utilities for concurrency control.

This module provides the `QueryLimiter` class for controlling concurrent
execution of heavy database queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import anyio
from anyio import to_thread

from codeintel.serving.context import ServingContext
from codeintel.serving.settings import ServingSettings

P = ParamSpec("P")
T = TypeVar("T")


class QueryLimiter:
    """Semaphore-based limiter for concurrent query execution.

    Prevent resource exhaustion from concurrent heavy queries by
    limiting the number of simultaneous database operations.

    Parameters
    ----------
    max_concurrent
        Maximum number of concurrent queries allowed.

    Examples
    --------
    >>> limiter = QueryLimiter(max_concurrent=2)
    >>> result = await limiter.run(kernel.query, request)
    """

    def __init__(self, max_concurrent: int) -> None:
        """Initialize the query limiter.

        Parameters
        ----------
        max_concurrent
            Maximum number of concurrent queries allowed.
        """
        self._sem = anyio.Semaphore(max_concurrent)
        self._max = max_concurrent

    @property
    def max_concurrent(self) -> int:
        """Return the maximum concurrent queries allowed.

        Returns
        -------
        int
            Maximum concurrent queries.
        """
        return self._max

    async def run(
        self,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute a synchronous function with concurrency limiting.

        Acquire the semaphore, then offload the function to a thread
        pool to avoid blocking the event loop.

        Parameters
        ----------
        fn
            Synchronous function to execute.
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        T
            Result from the function.
        """
        async with self._sem:
            return await to_thread.run_sync(
                lambda: fn(*args, **kwargs),
                abandon_on_cancel=True,
            )

    async def run_with_timeout(
        self,
        fn: Callable[P, T],
        timeout_s: float | None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute a synchronous function with an optional timeout.

        Parameters
        ----------
        fn
            Synchronous function to execute.
        timeout_s
            Optional timeout in seconds (None disables timeout).
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        T
            Result from the function.
        """
        async with self._sem:
            if timeout_s is None:
                return await to_thread.run_sync(
                    lambda: fn(*args, **kwargs),
                    abandon_on_cancel=True,
                )
            with anyio.fail_after(timeout_s):
                return await to_thread.run_sync(
                    lambda: fn(*args, **kwargs),
                    abandon_on_cancel=True,
                )

    async def run_async(
        self,
        coro_fn: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute an async function with concurrency limiting.

        Acquire the semaphore, then await the coroutine.

        Parameters
        ----------
        coro_fn
            Async function to execute.
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        T
            Result from the coroutine.
        """
        async with self._sem:
            return await coro_fn(*args, **kwargs)

    async def run_async_with_timeout(
        self,
        coro_fn: Callable[P, Awaitable[T]],
        timeout_s: float | None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute an async function with an optional timeout.

        Parameters
        ----------
        coro_fn
            Async function to execute.
        timeout_s
            Optional timeout in seconds (None disables timeout).
        *args
            Positional arguments for the coroutine.
        **kwargs
            Keyword arguments for the coroutine.

        Returns
        -------
        T
            Result from the coroutine.
        """
        async with self._sem:
            if timeout_s is None:
                return await coro_fn(*args, **kwargs)
            with anyio.fail_after(timeout_s):
                return await coro_fn(*args, **kwargs)


def query_limiter_from_settings(settings: ServingSettings) -> QueryLimiter:
    """Build a query limiter from serving settings.

    Returns
    -------
    QueryLimiter
        Configured limiter for query concurrency.
    """
    return QueryLimiter(max_concurrent=settings.mcp_max_concurrent_queries)


def export_limiter_from_settings(settings: ServingSettings) -> QueryLimiter:
    """Build an export limiter from serving settings.

    Returns
    -------
    QueryLimiter
        Configured limiter for export concurrency.
    """
    return QueryLimiter(max_concurrent=settings.mcp_max_concurrent_exports)


def query_limiter_from_context(context: ServingContext) -> QueryLimiter:
    """Build a query limiter from a serving context.

    Returns
    -------
    QueryLimiter
        Configured limiter for query concurrency.
    """
    return query_limiter_from_settings(context.settings)


def export_limiter_from_context(context: ServingContext) -> QueryLimiter:
    """Build an export limiter from a serving context.

    Returns
    -------
    QueryLimiter
        Configured limiter for export concurrency.
    """
    return export_limiter_from_settings(context.settings)


__all__ = [
    "QueryLimiter",
    "export_limiter_from_context",
    "export_limiter_from_settings",
    "query_limiter_from_context",
    "query_limiter_from_settings",
]
