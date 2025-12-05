"""Timing utilities for measuring execution duration.

This module provides lightweight utilities for measuring execution time
in a consistent manner across the codebase. It replaces ad-hoc
`time.perf_counter()` patterns with reusable, typed utilities.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar

T = TypeVar("T")


@dataclass
class TimingResult:
    """Result of a timed operation.

    Provide elapsed time in various units (nanoseconds, milliseconds, seconds)
    as computed properties based on the captured start and end times.

    Attributes
    ----------
    start_ns
        Start time in nanoseconds (from perf_counter_ns).
    end_ns
        End time in nanoseconds, or None if not yet stopped.

    Examples
    --------
    >>> result = TimingResult()
    >>> # ... do work ...
    >>> result.stop()
    >>> print(f"Took {result.elapsed_ms:.2f}ms")
    """

    start_ns: int = field(default_factory=time.perf_counter_ns)
    end_ns: int | None = None

    def stop(self) -> None:
        """Record the end time.

        Call this method when the operation being timed is complete.
        If already stopped, this method has no effect.
        """
        if self.end_ns is None:
            self.end_ns = time.perf_counter_ns()

    @property
    def is_stopped(self) -> bool:
        """Check if timing has been stopped.

        Returns
        -------
        bool
            True if stop() has been called.
        """
        return self.end_ns is not None

    @property
    def elapsed_ns(self) -> int:
        """Return elapsed time in nanoseconds.

        If timing has not been stopped, return the elapsed time up to now.

        Returns
        -------
        int
            Elapsed nanoseconds.
        """
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return end - self.start_ns

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds.

        Returns
        -------
        float
            Elapsed milliseconds.
        """
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Return elapsed time in seconds.

        Returns
        -------
        float
            Elapsed seconds.
        """
        return self.elapsed_ns / 1_000_000_000


@contextmanager
def timed() -> Iterator[TimingResult]:
    """Context manager for timing operations.

    Automatically start timing on entry and stop on exit. The TimingResult
    is yielded so elapsed time can be accessed after the block completes.

    Yields
    ------
    TimingResult
        A timing result object that tracks elapsed time.

    Examples
    --------
    >>> with timed() as t:
    ...     # do work
    ...     pass
    >>> print(f"Operation took {t.elapsed_ms:.2f}ms")
    """
    result = TimingResult()
    try:
        yield result
    finally:
        result.stop()


def measure_duration[T](
    fn: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> tuple[T, TimingResult]:
    """Execute a function and return the result with timing information.

    Parameters
    ----------
    fn
        The function to execute.
    *args
        Positional arguments to pass to the function.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    tuple[T, TimingResult]
        A tuple of (function result, timing result).

    Examples
    --------
    >>> def slow_function(x: int) -> int:
    ...     import time
    ...
    ...     time.sleep(0.1)
    ...     return x * 2
    >>> result, timing = measure_duration(slow_function, 5)
    >>> print(f"Result: {result}, Duration: {timing.elapsed_ms:.2f}ms")
    """
    with timed() as timing:
        result = fn(*args, **kwargs)
    return result, timing


def measure_duration_ms[T](
    fn: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> tuple[T, float]:
    """Execute a function and return the result with duration in milliseconds.

    This is a convenience wrapper around measure_duration that returns
    just the milliseconds value instead of the full TimingResult.

    Parameters
    ----------
    fn
        The function to execute.
    *args
        Positional arguments to pass to the function.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    tuple[T, float]
        A tuple of (function result, duration in milliseconds).

    Examples
    --------
    >>> def process(x: int) -> int:
    ...     return x * 2
    >>> result, duration_ms = measure_duration_ms(process, 5)
    >>> print(f"Result: {result}, Duration: {duration_ms:.2f}ms")
    """
    result, timing = measure_duration(fn, *args, **kwargs)
    return result, timing.elapsed_ms


def utc_now() -> datetime:
    """Return the current UTC datetime.

    Use this function instead of ``datetime.now(tz=UTC)`` for consistency
    across the codebase. This makes timestamps easier to search for and
    provides a single point for any future datetime behavior changes.

    Returns
    -------
    datetime
        Current datetime in UTC timezone.

    Examples
    --------
    >>> from codeintel.core.execution.timing import utc_now
    >>> now = utc_now()
    >>> now.tzinfo is UTC
    True
    """
    return datetime.now(tz=UTC)


__all__ = [
    "TimingResult",
    "measure_duration",
    "measure_duration_ms",
    "timed",
    "utc_now",
]
