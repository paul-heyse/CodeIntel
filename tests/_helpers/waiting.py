"""Eventually helpers for polling-based tests."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


def eventually[T](
    assert_fn: Callable[[], T],
    *,
    timeout_s: float = 5.0,
    interval_s: float = 0.1,
    retry_on: Sequence[type[Exception]] = (AssertionError,),
    message: str | None = None,
) -> T:
    """Retry an assertion-style callable until it succeeds or times out.

    Parameters
    ----------
    assert_fn
        Callable that should succeed without raising.
    timeout_s
        Maximum time to wait before failing.
    interval_s
        Sleep interval between attempts.
    retry_on
        Exception types that trigger retries.
    message
        Optional message prefix for timeout failures.

    Returns
    -------
    T_co
        Value returned by assert_fn when it succeeds.

    Raises
    ------
    AssertionError
        When the timeout is exceeded.
    """
    _validate_wait_params(timeout_s=timeout_s, interval_s=interval_s)
    last_exc: BaseException | None = None
    deadline = time.monotonic() + timeout_s
    retry_types = tuple(retry_on)

    while time.monotonic() <= deadline:
        try:
            return assert_fn()
        except Exception as exc:
            if not isinstance(exc, retry_types):
                raise
            last_exc = exc
        time.sleep(interval_s)

    detail = f" after {timeout_s:.2f}s"
    prefix = message or "eventually timed out"
    if last_exc is None:
        message = f"{prefix}{detail}"
        raise AssertionError(message)
    message = f"{prefix}{detail}: {last_exc}"
    raise AssertionError(message) from last_exc


async def eventually_async[T](
    assert_fn: Callable[[], Awaitable[T]],
    *,
    timeout_s: float = 5.0,
    interval_s: float = 0.1,
    retry_on: Sequence[type[Exception]] = (AssertionError,),
    message: str | None = None,
) -> T:
    """Retry an async assertion-style callable until it succeeds or times out.

    Parameters
    ----------
    assert_fn
        Async callable that should succeed without raising.
    timeout_s
        Maximum time to wait before failing.
    interval_s
        Sleep interval between attempts.
    retry_on
        Exception types that trigger retries.
    message
        Optional message prefix for timeout failures.

    Returns
    -------
    T_co
        Value returned by assert_fn when it succeeds.

    Raises
    ------
    AssertionError
        When the timeout is exceeded.
    """
    _validate_wait_params(timeout_s=timeout_s, interval_s=interval_s)
    last_exc: BaseException | None = None
    deadline = time.monotonic() + timeout_s
    retry_types = tuple(retry_on)

    while time.monotonic() <= deadline:
        try:
            return await assert_fn()
        except Exception as exc:
            if not isinstance(exc, retry_types):
                raise
            last_exc = exc
        await asyncio.sleep(interval_s)

    detail = f" after {timeout_s:.2f}s"
    prefix = message or "eventually timed out"
    if last_exc is None:
        message = f"{prefix}{detail}"
        raise AssertionError(message)
    message = f"{prefix}{detail}: {last_exc}"
    raise AssertionError(message) from last_exc


def _validate_wait_params(*, timeout_s: float, interval_s: float) -> None:
    if timeout_s <= 0:
        message = "timeout_s must be positive"
        raise ValueError(message)
    if interval_s <= 0:
        message = "interval_s must be positive"
        raise ValueError(message)
    if interval_s > timeout_s:
        message = "interval_s must be <= timeout_s"
        raise ValueError(message)


__all__ = ["eventually", "eventually_async"]
