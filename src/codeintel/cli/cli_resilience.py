"""Resilience and retry infrastructure for CLI operations.

Provides retry logic, circuit breakers, and fallback mechanisms
for operations that may experience transient failures.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

LOG = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open.

    Parameters
    ----------
    message
        Error message.
    retry_after
        Seconds until circuit may close.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        """Initialize circuit open error."""
        super().__init__(message)
        self.retry_after = retry_after


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

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

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)  # noqa: S311

        return max(0, delay)


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing repeated failures.

    Parameters
    ----------
    failure_threshold
        Number of failures before opening circuit.
    recovery_timeout
        Seconds before attempting recovery.
    half_open_max_calls
        Max calls in half-open state before deciding.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state.

        Returns
        -------
        CircuitState
            Current state.
        """
        is_open = self._state == CircuitState.OPEN
        timeout_passed = time.monotonic() - self._last_failure_time >= self.recovery_timeout
        if is_open and timeout_passed:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            # Check if we've recovered (enough successful half-open calls)
            recovered = self._half_open_calls >= self.half_open_max_calls
            if recovered:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery, reopen
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns
        -------
        bool
            True if request is allowed.

        Raises
        ------
        CircuitOpenError
            If circuit is open.
        """
        state = self.state  # This may transition from OPEN to HALF_OPEN

        if state == CircuitState.OPEN:
            retry_after = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            msg = "Circuit breaker is open"
            safe_retry_after = retry_after if retry_after > 0.0 else 0.0
            raise CircuitOpenError(msg, retry_after=safe_retry_after)

        return True


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
    """

    attempt: int
    exception: Exception | None
    delay: float


def _handle_retry(
    policy: RetryPolicy,
    attempt: int,
    exc: Exception,
    on_retry: Callable[[RetryContext], None] | None,
) -> None:
    """Handle retry logic for a failed attempt.

    Parameters
    ----------
    policy
        Retry policy.
    attempt
        Current attempt number.
    exc
        Exception that occurred.
    on_retry
        Optional retry callback.
    """
    delay = policy.calculate_delay(attempt)

    if on_retry:
        ctx = RetryContext(attempt=attempt, exception=exc, delay=delay)
        on_retry(ctx)

    LOG.warning(
        "Retrying after error",
        extra={
            "attempt": attempt + 1,
            "max_attempts": policy.max_attempts,
            "delay": delay,
            "error": str(exc),
        },
    )
    time.sleep(delay)


def _log_max_retries(policy: RetryPolicy, exc: Exception) -> None:
    """Log when max retries are exceeded.

    Parameters
    ----------
    policy
        Retry policy.
    exc
        Final exception.
    """
    LOG.warning(
        "Max retries exceeded",
        extra={
            "attempts": policy.max_attempts,
            "error": str(exc),
        },
    )


def _handle_exception(
    policy: RetryPolicy,
    attempt: int,
    exc: Exception,
    on_retry: Callable[[RetryContext], None] | None,
    circuit_breaker: CircuitBreaker | None,
) -> None:
    """Handle an exception during retry.

    Parameters
    ----------
    policy
        Retry policy.
    attempt
        Current attempt number.
    exc
        Exception that occurred.
    on_retry
        Optional retry callback.
    circuit_breaker
        Optional circuit breaker.
    """
    if circuit_breaker:
        circuit_breaker.record_failure()

    has_retries = attempt < policy.max_attempts - 1
    if has_retries:
        _handle_retry(policy, attempt, exc, on_retry)
    else:
        _log_max_retries(policy, exc)


def with_retry(
    policy: RetryPolicy | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    on_retry: Callable[[RetryContext], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a function to add retry logic.

    Parameters
    ----------
    policy
        Retry policy (uses default if None).
    circuit_breaker
        Optional circuit breaker.
    on_retry
        Callback invoked before each retry.

    Returns
    -------
    Callable
        Decorated function.

    Example
    -------
    >>> @with_retry(RetryPolicy(max_attempts=3))
    ... def fetch_data(url: str) -> dict:
    ...     response = requests.get(url)
    ...     response.raise_for_status()
    ...     return response.json()
    """
    effective_policy = policy or RetryPolicy()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if circuit_breaker:
                circuit_breaker.allow_request()

            last_exception: Exception | None = None

            for attempt in range(effective_policy.max_attempts):
                try:
                    result = func(*args, **kwargs)
                except effective_policy.retryable_exceptions as e:
                    last_exception = e
                    _handle_exception(effective_policy, attempt, e, on_retry, circuit_breaker)
                else:
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result

            if last_exception is None:
                msg = "Retry loop completed without exception or result"
                raise RuntimeError(msg)
            raise last_exception

        return wrapper

    return decorator


class RetryMiddleware:
    """Middleware that adds retry logic to operations.

    Parameters
    ----------
    policy
        Default retry policy.
    circuit_breaker
        Shared circuit breaker.
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize retry middleware."""
        self._policy = policy or RetryPolicy()
        self._circuit_breaker = circuit_breaker

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Check circuit breaker before operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.
        """
        _ = op_id, params  # Use parameters to satisfy linter
        if self._circuit_breaker:
            self._circuit_breaker.allow_request()
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record success after operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        _ = op_id, result, context  # Use parameters to satisfy linter
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record failure after operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        _ = op_id, exc, context  # Use parameters to satisfy linter
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()


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


__all__ = [
    "DEFAULT_NETWORK_POLICY",
    "DEFAULT_STORAGE_POLICY",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "RetryContext",
    "RetryMiddleware",
    "RetryPolicy",
    "RetryableError",
    "with_retry",
]
