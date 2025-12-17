"""MCP runtime utilities for concurrency control.

This module provides the `QueryLimiter` class for controlling concurrent
execution of heavy database queries.
"""

from __future__ import annotations

from typing import TypeVar

import anyio
from anyio import to_thread

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

    async def run(self, fn: object, *args: object, **kwargs: object) -> object:
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
        object
            Result from the function.
        """
        async with self._sem:
            return await to_thread.run_sync(lambda: fn(*args, **kwargs))  # type: ignore[operator]

    async def run_async(self, coro_fn: object, *args: object, **kwargs: object) -> object:
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
        object
            Result from the coroutine.
        """
        async with self._sem:
            return await coro_fn(*args, **kwargs)  # type: ignore[operator]


__all__ = ["QueryLimiter"]
