"""Retry logic for CLI operations.

Provide retry policies with exponential backoff and jitter,
supporting both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from codeintel.cli.core import CliResult

LOG = logging.getLogger(__name__)

# Use cryptographically secure RNG for jitter (satisfies S311)
_SECURE_RANDOM = secrets.SystemRandom()

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior.

    Support both synchronous and asynchronous operations with
    exponential backoff and jitter.

    Parameters
    ----------
    max_attempts
        Maximum number of attempts (including initial).
    initial_delay
        Initial delay between retries in seconds.
    max_delay
        Maximum delay between retries in seconds.
    backoff_factor
        Multiplier for exponential backoff.
    jitter
        Random jitter factor (0.0 to 1.0).
    retryable_exceptions
        Exception types that should trigger retry.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError,)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Parameters
        ----------
        attempt
            Attempt number (0-indexed).

        Returns
        -------
        float
            Delay in seconds.
        """
        delay = self.initial_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)

        jitter_range = delay * self.jitter
        delay += _SECURE_RANDOM.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    def is_retryable(self, exc: Exception) -> bool:
        """Check if exception is retryable.

        Parameters
        ----------
        exc
            Exception to check.

        Returns
        -------
        bool
            True if retryable.
        """
        return isinstance(exc, self.retryable_exceptions)


# Default policies for common scenarios
DEFAULT_NETWORK_POLICY = RetryPolicy(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, RetryableError),
)

DEFAULT_STORAGE_POLICY = RetryPolicy(
    max_attempts=2,
    initial_delay=0.5,
    max_delay=5.0,
    backoff_factor=2.0,
    retryable_exceptions=(RetryableError,),
)


@dataclass(frozen=True)
class RetryContext:
    """Context passed to retry callbacks.

    Parameters
    ----------
    attempt
        Current attempt number (0-indexed).
    exception
        Exception that triggered retry (if any).
    delay
        Delay before next attempt.
    operation_id
        Optional operation identifier.
    """

    attempt: int
    exception: Exception | None
    delay: float
    operation_id: str = ""


RetryCallback = Callable[[RetryContext], None]
"""Callback type for retry events."""


@dataclass(frozen=True)
class RetryOptions:
    """Options for retry execution.

    Parameters
    ----------
    operation_id
        Operation identifier for logging.
    on_retry
        Callback for retry events.
    circuit_breaker
        Optional circuit breaker.
    """

    operation_id: str = ""
    on_retry: RetryCallback | None = None
    circuit_breaker: object | None = None  # Avoid circular import


def _handle_retry_delay(
    policy: RetryPolicy,
    attempt: int,
    exc: Exception,
    options: RetryOptions,
) -> None:
    """Handle delay and callbacks for a retry attempt.

    Parameters
    ----------
    policy
        Retry policy.
    attempt
        Current attempt number (0-indexed).
    exc
        Exception that triggered retry.
    options
        Retry options.
    """
    delay = policy.calculate_delay(attempt)
    LOG.warning(
        "Operation %s failed (attempt %d/%d), retrying in %.1fs: %s",
        options.operation_id,
        attempt + 1,
        policy.max_attempts,
        delay,
        exc,
    )
    if options.on_retry:
        ctx = RetryContext(
            attempt=attempt,
            exception=exc,
            delay=delay,
            operation_id=options.operation_id,
        )
        options.on_retry(ctx)
    time.sleep(delay)


async def _handle_retry_delay_async(
    policy: RetryPolicy,
    attempt: int,
    exc: Exception,
    options: RetryOptions,
) -> None:
    """Handle delay and callbacks for a retry attempt (async).

    Parameters
    ----------
    policy
        Retry policy.
    attempt
        Current attempt number (0-indexed).
    exc
        Exception that triggered retry.
    options
        Retry options.
    """
    delay = policy.calculate_delay(attempt)
    LOG.warning(
        "Operation %s failed (attempt %d/%d), retrying in %.1fs: %s",
        options.operation_id,
        attempt + 1,
        policy.max_attempts,
        delay,
        exc,
    )
    if options.on_retry:
        ctx = RetryContext(
            attempt=attempt,
            exception=exc,
            delay=delay,
            operation_id=options.operation_id,
        )
        options.on_retry(ctx)
    await asyncio.sleep(delay)


def _log_max_retries_exceeded(policy: RetryPolicy, operation_id: str) -> None:
    """Log when max retries are exceeded.

    Parameters
    ----------
    policy
        Retry policy.
    operation_id
        Operation identifier.
    """
    LOG.warning(
        "Operation %s max retries exceeded after %d attempts",
        operation_id,
        policy.max_attempts,
    )


def _raise_if_no_result(last_exception: Exception | None) -> None:
    """Raise appropriate exception when retry loop completes.

    Parameters
    ----------
    last_exception
        Last exception from retry attempts.

    Raises
    ------
    RuntimeError
        If no exception occurred or no result was produced.

    Notes
    -----
    If ``last_exception`` is not None, it will be re-raised.
    """
    if last_exception is not None:
        raise last_exception

    msg = "Retry loop completed without exception or result"
    raise RuntimeError(msg)


def execute_with_retry[T](
    handler: Callable[..., T],
    params: dict[str, Any],
    policy: RetryPolicy,
    options: RetryOptions | None = None,
) -> T:
    """Execute handler with retry policy (synchronous).

    Parameters
    ----------
    handler
        Handler function to execute.
    params
        Keyword arguments for handler.
    policy
        Retry policy.
    options
        Optional retry options.

    Returns
    -------
    T
        Handler result.

    Raises
    ------
    RuntimeError
        If retry loop completes without a result (internal error).

    Notes
    -----
    If all retries are exhausted, the last retryable exception is re-raised.
    Non-retryable exceptions propagate immediately without retry.
    """
    opts = options or RetryOptions()

    if opts.circuit_breaker is not None:
        opts.circuit_breaker.allow_request()  # type: ignore[union-attr]

    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = handler(**params)
        except retryable as e:
            last_exception = e
            if opts.circuit_breaker is not None:
                opts.circuit_breaker.record_failure()  # type: ignore[union-attr]

            if attempt < policy.max_attempts - 1:
                _handle_retry_delay(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            if opts.circuit_breaker is not None:
                opts.circuit_breaker.record_success()  # type: ignore[union-attr]
            return result

    _raise_if_no_result(last_exception)
    msg = "Unreachable"
    raise RuntimeError(msg)


def execute_cli_with_retry[T](
    handler: Callable[..., CliResult[T]],
    params: dict[str, Any],
    policy: RetryPolicy,
    options: RetryOptions | None = None,
) -> CliResult[T]:
    """Execute CLI handler with retry policy.

    Parameters
    ----------
    handler
        CLI handler function returning CliResult.
    params
        Handler parameters.
    policy
        Retry policy.
    options
        Optional retry options.

    Returns
    -------
    CliResult[T]
        Handler result.

    Raises
    ------
    RuntimeError
        If retry loop completes without a result (internal error).

    Notes
    -----
    If all retries are exhausted, the last retryable exception is re-raised.
    Non-retryable exceptions propagate immediately without retry.
    """
    opts = options or RetryOptions()
    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = handler(**params)
        except retryable as e:
            last_exception = e
            if attempt < policy.max_attempts - 1:
                _handle_retry_delay(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            return result

    _raise_if_no_result(last_exception)
    msg = "Unreachable"
    raise RuntimeError(msg)


async def execute_with_retry_async[T](
    handler: Callable[..., Awaitable[T]],
    params: dict[str, Any],
    policy: RetryPolicy,
    options: RetryOptions | None = None,
) -> T:
    """Execute async handler with retry policy.

    Parameters
    ----------
    handler
        Async handler function to execute.
    params
        Keyword arguments for handler.
    policy
        Retry policy.
    options
        Optional retry options.

    Returns
    -------
    T
        Handler result.

    Raises
    ------
    RuntimeError
        If retry loop completes without a result (internal error).

    Notes
    -----
    If all retries are exhausted, the last retryable exception is re-raised.
    Non-retryable exceptions propagate immediately without retry.
    """
    opts = options or RetryOptions()

    if opts.circuit_breaker is not None:
        opts.circuit_breaker.allow_request()  # type: ignore[union-attr]

    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = await handler(**params)
        except retryable as e:
            last_exception = e
            if opts.circuit_breaker is not None:
                opts.circuit_breaker.record_failure()  # type: ignore[union-attr]

            if attempt < policy.max_attempts - 1:
                await _handle_retry_delay_async(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            if opts.circuit_breaker is not None:
                opts.circuit_breaker.record_success()  # type: ignore[union-attr]
            return result

    _raise_if_no_result(last_exception)
    msg = "Unreachable"
    raise RuntimeError(msg)


async def execute_cli_with_retry_async[T](
    handler: Callable[..., Awaitable[CliResult[T]]],
    params: dict[str, Any],
    policy: RetryPolicy,
    options: RetryOptions | None = None,
) -> CliResult[T]:
    """Execute async CLI handler with retry policy.

    Parameters
    ----------
    handler
        Async CLI handler function returning CliResult.
    params
        Handler parameters.
    policy
        Retry policy.
    options
        Optional retry options.

    Returns
    -------
    CliResult[T]
        Handler result.

    Raises
    ------
    RuntimeError
        If retry loop completes without a result (internal error).

    Notes
    -----
    If all retries are exhausted, the last retryable exception is re-raised.
    Non-retryable exceptions propagate immediately without retry.
    """
    opts = options or RetryOptions()
    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = await handler(**params)
        except retryable as e:
            last_exception = e
            if attempt < policy.max_attempts - 1:
                await _handle_retry_delay_async(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            return result

    _raise_if_no_result(last_exception)
    msg = "Unreachable"
    raise RuntimeError(msg)


def with_retry(
    policy: RetryPolicy | None = None,
    options: RetryOptions | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a function to add retry logic.

    Parameters
    ----------
    policy
        Retry policy (uses default if None).
    options
        Retry options (circuit breaker, callbacks).

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> @with_retry(RetryPolicy(max_attempts=3))
    ... def fetch_data(url: str) -> dict:
    ...     response = requests.get(url)
    ...     response.raise_for_status()
    ...     return response.json()
    """
    effective_policy = policy or RetryPolicy()
    effective_options = options or RetryOptions()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            captured_args = args
            captured_kwargs = kwargs

            def invoke(**params: object) -> T:
                # Merge captured kwargs with params (params override)
                merged = dict(captured_kwargs)
                merged.update(params)
                return func(*captured_args, **merged)

            return execute_with_retry(
                invoke,
                dict(kwargs),
                effective_policy,
                effective_options,
            )

        return wrapper

    return decorator


__all__ = [
    "DEFAULT_NETWORK_POLICY",
    "DEFAULT_STORAGE_POLICY",
    "RetryCallback",
    "RetryContext",
    "RetryOptions",
    "RetryPolicy",
    "RetryableError",
    "execute_cli_with_retry",
    "execute_cli_with_retry_async",
    "execute_with_retry",
    "execute_with_retry_async",
    "with_retry",
]
