"""Tenacity-based retry policies for plugin execution.

This module provides a standardized, configurable approach to retrying
transient failures using the tenacity library. It replaces ad-hoc retry
loops with well-tested, feature-rich retry logic.

The module provides:
- RetryPolicy: Configurable retry behavior specification
- Pre-configured policies for common use cases
- Decorators and context managers for different retry patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)

from codeintel.core.execution.errors import PLUGIN_CATCHABLE_ERRORS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tenacity import (
        RetryCallState,
    )
    from tenacity.stop import stop_base

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configurable retry policy using tenacity under the hood.

    Define retry behavior including:
    - Maximum attempts and time limits
    - Backoff strategy (exponential with or without jitter)
    - Exception types that trigger retries

    Attributes
    ----------
    max_attempts
        Maximum number of retry attempts (including initial attempt).
    max_delay_s
        Maximum total time to spend retrying (deadline).
    backoff_multiplier
        Multiplier for exponential backoff calculation.
    backoff_max_s
        Maximum wait time between retries.
    retryable_exceptions
        Exception types that should trigger retries.
    use_jitter
        Whether to add randomization to backoff (recommended).
    log_retries
        Whether to log retry attempts.

    Examples
    --------
    >>> policy = RetryPolicy(max_attempts=3, use_jitter=True)
    >>> @policy.as_decorator()
    ... def unstable_operation():
    ...     # May fail transiently
    ...     pass

    >>> # Or use block-style retrying
    >>> for attempt in policy.create_retrying():
    ...     with attempt:
    ...         unstable_operation()
    """

    max_attempts: int = 3
    max_delay_s: float = 30.0
    backoff_multiplier: float = 0.5
    backoff_max_s: float = 10.0
    retryable_exceptions: tuple[type[Exception], ...] = PLUGIN_CATCHABLE_ERRORS
    use_jitter: bool = True
    log_retries: bool = True

    def get_wait_strategy(self) -> wait_exponential | wait_random_exponential:
        """Return the appropriate wait strategy based on configuration.

        Returns
        -------
        wait_exponential | wait_random_exponential
            Tenacity wait strategy instance.
        """
        if self.use_jitter:
            return wait_random_exponential(
                multiplier=self.backoff_multiplier,
                max=self.backoff_max_s,
            )
        return wait_exponential(
            multiplier=self.backoff_multiplier,
            max=self.backoff_max_s,
        )

    def get_stop_strategy(self) -> stop_base:
        """Return the stop strategy combining attempts and delay limits.

        Returns
        -------
        stop_base
            Tenacity stop strategy (combined via | operator).
        """
        return stop_after_attempt(self.max_attempts) | stop_after_delay(self.max_delay_s)

    def get_before_sleep(self) -> Callable[[RetryCallState], None] | None:
        """Return the before_sleep callback for logging.

        Returns
        -------
        Callable[[RetryCallState], None] | None
            Callback function for logging, or None if disabled.
        """
        if self.log_retries:
            return before_sleep_log(log, logging.WARNING, exc_info=True)
        return None

    def create_retrying(self) -> Retrying:
        """Create a tenacity Retrying controller for block-style retries.

        Use this for retrying multi-step operations as a unit.

        Returns
        -------
        Retrying
            A tenacity Retrying instance configured with this policy.

        Examples
        --------
        >>> policy = RetryPolicy(max_attempts=3)
        >>> for attempt in policy.create_retrying():
        ...     with attempt:
        ...         step1()
        ...         step2()  # Both steps retry together
        """
        return Retrying(
            stop=self.get_stop_strategy(),
            wait=self.get_wait_strategy(),
            retry=retry_if_exception_type(self.retryable_exceptions),
            before_sleep=self.get_before_sleep(),
            reraise=True,
        )

    def create_async_retrying(self) -> AsyncRetrying:
        """Create a tenacity AsyncRetrying controller for async block-style retries.

        Returns
        -------
        AsyncRetrying
            A tenacity AsyncRetrying instance configured with this policy.

        Examples
        --------
        >>> policy = RetryPolicy(max_attempts=3)
        >>> async for attempt in policy.create_async_retrying():
        ...     with attempt:
        ...         await async_operation()
        """
        return AsyncRetrying(
            stop=self.get_stop_strategy(),
            wait=self.get_wait_strategy(),
            retry=retry_if_exception_type(self.retryable_exceptions),
            before_sleep=self.get_before_sleep(),
            reraise=True,
        )

    def as_decorator(self) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a tenacity decorator configured with this policy.

        The decorator works for both sync and async functions.

        Returns
        -------
        Callable
            A decorator that adds retry behavior to functions.

        Examples
        --------
        >>> policy = RetryPolicy(max_attempts=5)
        >>> @policy.as_decorator()
        ... def flaky_function():
        ...     pass
        """
        return retry(
            stop=self.get_stop_strategy(),
            wait=self.get_wait_strategy(),
            retry=retry_if_exception_type(self.retryable_exceptions),
            before_sleep=self.get_before_sleep(),
            reraise=True,
        )

    def with_max_attempts(self, max_attempts: int) -> RetryPolicy:
        """Create a copy with different max_attempts.

        Parameters
        ----------
        max_attempts
            New maximum attempts value.

        Returns
        -------
        RetryPolicy
            New policy with updated max_attempts.
        """
        return RetryPolicy(
            max_attempts=max_attempts,
            max_delay_s=self.max_delay_s,
            backoff_multiplier=self.backoff_multiplier,
            backoff_max_s=self.backoff_max_s,
            retryable_exceptions=self.retryable_exceptions,
            use_jitter=self.use_jitter,
            log_retries=self.log_retries,
        )

    def with_exceptions(self, exceptions: tuple[type[Exception], ...]) -> RetryPolicy:
        """Create a copy with different retryable exceptions.

        Parameters
        ----------
        exceptions
            New tuple of retryable exception types.

        Returns
        -------
        RetryPolicy
            New policy with updated exceptions.
        """
        return RetryPolicy(
            max_attempts=self.max_attempts,
            max_delay_s=self.max_delay_s,
            backoff_multiplier=self.backoff_multiplier,
            backoff_max_s=self.backoff_max_s,
            retryable_exceptions=exceptions,
            use_jitter=self.use_jitter,
            log_retries=self.log_retries,
        )


# =============================================================================
# Pre-configured policies for common use cases
# =============================================================================

#: Default policy for plugin execution with moderate retries
PLUGIN_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    max_delay_s=30.0,
    backoff_multiplier=0.5,
    backoff_max_s=5.0,
)

#: Policy for network operations with more aggressive retries
NETWORK_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    max_delay_s=60.0,
    backoff_multiplier=1.0,
    backoff_max_s=20.0,
    retryable_exceptions=(TimeoutError, ConnectionError, OSError),
)

#: Policy that disables retries (single attempt only)
NO_RETRY_POLICY = RetryPolicy(
    max_attempts=1,
    max_delay_s=0.0,
)

#: Policy for database operations
DATABASE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    max_delay_s=15.0,
    backoff_multiplier=0.25,
    backoff_max_s=3.0,
)


# =============================================================================
# Helper functions
# =============================================================================


_RETRY_EXHAUSTED_MSG = "Retry exhausted without result or exception"


def with_retry[T](
    policy: RetryPolicy,
    fn: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    """Execute a function with retry policy using block-style retrying.

    Parameters
    ----------
    policy
        Retry policy to use.
    fn
        Function to execute.
    *args
        Positional arguments to pass to the function.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    T
        The function's return value.

    Raises
    ------
    RuntimeError
        If retry logic fails unexpectedly (should not occur with reraise=True).

    Examples
    --------
    >>> def flaky_call():
    ...     import random
    ...
    ...     if random.random() < 0.5:
    ...         raise ValueError("Transient failure")
    ...     return "success"
    >>> result = with_retry(PLUGIN_RETRY_POLICY, flaky_call)
    """
    for attempt in policy.create_retrying():
        with attempt:
            return fn(*args, **kwargs)
    # Should not reach here due to reraise=True, but satisfy type checker
    raise RuntimeError(_RETRY_EXHAUSTED_MSG)


async def with_retry_async[T](
    policy: RetryPolicy,
    fn: Callable[..., Awaitable[T]],
    *args: object,
    **kwargs: object,
) -> T:
    """Execute an async function with retry policy.

    Parameters
    ----------
    policy
        Retry policy to use.
    fn
        Async function to execute.
    *args
        Positional arguments to pass to the function.
    **kwargs
        Keyword arguments to pass to the function.

    Returns
    -------
    T
        The function's return value.

    Raises
    ------
    RuntimeError
        If retry logic fails unexpectedly (should not occur with reraise=True).
    """
    async for attempt in policy.create_async_retrying():
        with attempt:
            return await fn(*args, **kwargs)
    # Should not reach here due to reraise=True, but satisfy type checker
    raise RuntimeError(_RETRY_EXHAUSTED_MSG)


def get_retry_policy_for_retryable(
    max_retries: int,
    retry_backoff_ms: int,
    retryable_exceptions: tuple[type[Exception], ...],
) -> RetryPolicy:
    """Create a RetryPolicy from retryable plugin attributes.

    This is a helper function for the RetryablePlugin trait to avoid
    circular imports.

    Parameters
    ----------
    max_retries
        Maximum retry attempts.
    retry_backoff_ms
        Backoff time in milliseconds.
    retryable_exceptions
        Exception types to retry.

    Returns
    -------
    RetryPolicy
        Configured retry policy.
    """
    return RetryPolicy(
        max_attempts=max_retries,
        backoff_multiplier=retry_backoff_ms / 1000,
        retryable_exceptions=retryable_exceptions,
    )


__all__ = [
    "DATABASE_RETRY_POLICY",
    "NETWORK_RETRY_POLICY",
    "NO_RETRY_POLICY",
    "PLUGIN_CATCHABLE_ERRORS",
    "PLUGIN_RETRY_POLICY",
    "RetryError",
    "RetryPolicy",
    "get_retry_policy_for_retryable",
    "with_retry",
    "with_retry_async",
]
