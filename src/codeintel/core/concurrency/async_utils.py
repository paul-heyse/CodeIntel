"""Async utilities.

This module provides utilities for async/sync interop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine


def run_sync[T](coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine synchronously.

    Parameters
    ----------
    coro
        Coroutine to run.

    Returns
    -------
    T
        Coroutine result.

    Examples
    --------
    >>> async def fetch_data() -> str:
    ...     return "data"
    >>> result = run_sync(fetch_data())
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)


__all__ = [
    "run_sync",
]
