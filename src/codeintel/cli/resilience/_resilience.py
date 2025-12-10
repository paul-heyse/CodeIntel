"""Unified resilience infrastructure for CLI operations.

Provide retry logic, circuit breakers, and resilience middleware
supporting both synchronous and asynchronous operations.

This module consolidates and unifies:
- Retry policies with exponential backoff
- Circuit breaker pattern for failure protection
- Middleware integration for automatic resilience
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from codeintel.cli.errors import ProblemDetail
from codeintel.cli.execution.middleware import Middleware
from codeintel.cli.core import CliResult

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)

# Use cryptographically secure RNG for jitter (satisfies S311)
_SECURE_RANDOM = secrets.SystemRandom()

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


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


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


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

    @property
    def failure_count(self) -> int:
        """Get current failure count.

        Returns
        -------
        int
            Number of recorded failures.
        """
        return self._failure_count

    @property
    def last_failure_time(self) -> float:
        """Get timestamp of last failure.

        Returns
        -------
        float
            Last failure timestamp (monotonic).
        """
        return self._last_failure_time

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            recovered = self._half_open_calls >= self.half_open_max_calls
            if recovered:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        should_open = (
            self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold
        )
        if should_open:
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
        state = self.state

        if state == CircuitState.OPEN:
            retry_after = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            msg = "Circuit breaker is open"
            safe_retry_after = max(0.0, retry_after)
            raise CircuitOpenError(msg, retry_after=safe_retry_after)

        return True

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0


# =============================================================================
# Retry Policy
# =============================================================================


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


# =============================================================================
# Retry Context and Callbacks
# =============================================================================


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
    circuit_breaker: CircuitBreaker | None = None


# =============================================================================
# Retry Helper Functions
# =============================================================================


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


# =============================================================================
# Synchronous Retry Execution
# =============================================================================


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

    if opts.circuit_breaker:
        opts.circuit_breaker.allow_request()

    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = handler(**params)
        except retryable as e:
            last_exception = e
            if opts.circuit_breaker:
                opts.circuit_breaker.record_failure()

            if attempt < policy.max_attempts - 1:
                _handle_retry_delay(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            if opts.circuit_breaker:
                opts.circuit_breaker.record_success()
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


# =============================================================================
# Asynchronous Retry Execution
# =============================================================================


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

    if opts.circuit_breaker:
        opts.circuit_breaker.allow_request()

    last_exception: Exception | None = None
    retryable = policy.retryable_exceptions

    for attempt in range(policy.max_attempts):
        try:
            result = await handler(**params)
        except retryable as e:
            last_exception = e
            if opts.circuit_breaker:
                opts.circuit_breaker.record_failure()

            if attempt < policy.max_attempts - 1:
                await _handle_retry_delay_async(policy, attempt, e, opts)
            else:
                _log_max_retries_exceeded(policy, opts.operation_id)
        else:
            if opts.circuit_breaker:
                opts.circuit_breaker.record_success()
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


# =============================================================================
# Retry Decorator
# =============================================================================


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


# =============================================================================
# Circuit Breaker Registry
# =============================================================================


@dataclass
class CircuitBreakerStatus:
    """Status of a circuit breaker.

    Parameters
    ----------
    state
        Current state (closed, open, half_open).
    failure_count
        Number of recorded failures.
    last_failure_time
        Timestamp of last failure.
    """

    state: str
    failure_count: int
    last_failure_time: float


@dataclass
class ResilienceConfig:
    """Configuration for resilience behavior.

    Parameters
    ----------
    default_retry_policy
        Default retry policy for retryable operations.
    circuit_breaker_enabled
        Enable circuit breakers.
    circuit_failure_threshold
        Failures before circuit opens.
    circuit_recovery_timeout
        Seconds before attempting recovery.
    """

    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0


class CircuitBreakerRegistry:
    """Registry of circuit breakers by operation category.

    Maintain separate circuit breakers for different categories
    so failures in one category don't affect others.

    Parameters
    ----------
    config
        Resilience configuration.
    """

    def __init__(self, config: ResilienceConfig | None = None) -> None:
        """Initialize registry."""
        self._config = config or ResilienceConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    @property
    def config(self) -> ResilienceConfig:
        """Get configuration.

        Returns
        -------
        ResilienceConfig
            Current configuration.
        """
        return self._config

    def get_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for key.

        Parameters
        ----------
        key
            Circuit breaker key (usually operation category).

        Returns
        -------
        CircuitBreaker
            Circuit breaker instance.
        """
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                failure_threshold=self._config.circuit_failure_threshold,
                recovery_timeout=self._config.circuit_recovery_timeout,
            )
        return self._breakers[key]

    def get_status(self) -> dict[str, CircuitBreakerStatus]:
        """Get status of all circuit breakers.

        Returns
        -------
        dict[str, CircuitBreakerStatus]
            Status by key.
        """
        return {
            key: CircuitBreakerStatus(
                state=breaker.state.value,
                failure_count=breaker.failure_count,
                last_failure_time=breaker.last_failure_time,
            )
            for key, breaker in self._breakers.items()
        }

    def reset(self, key: str | None = None) -> None:
        """Reset circuit breaker(s).

        Parameters
        ----------
        key
            Specific key to reset, or None for all.
        """
        if key is not None:
            if key in self._breakers:
                self._breakers[key].reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()

    def clear(self, key: str | None = None) -> None:
        """Remove circuit breaker(s) from registry.

        Parameters
        ----------
        key
            Specific key to remove, or None for all.
        """
        if key is not None:
            self._breakers.pop(key, None)
        else:
            self._breakers.clear()


# =============================================================================
# Global Registry Management
# =============================================================================


class _GlobalRegistry:
    """Singleton manager for the global circuit breaker registry.

    This class encapsulates the global registry to avoid using
    global statement directly.
    """

    _instance: CircuitBreakerRegistry | None = None

    @classmethod
    def get(cls) -> CircuitBreakerRegistry:
        """Get or create the global registry.

        Returns
        -------
        CircuitBreakerRegistry
            Global registry.
        """
        if cls._instance is None:
            cls._instance = CircuitBreakerRegistry()
        return cls._instance

    @classmethod
    def configure(cls, config: ResilienceConfig) -> CircuitBreakerRegistry:
        """Configure the global registry.

        Parameters
        ----------
        config
            Resilience configuration.

        Returns
        -------
        CircuitBreakerRegistry
            Configured registry.
        """
        cls._instance = CircuitBreakerRegistry(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global registry to None (for testing)."""
        cls._instance = None


def get_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns
    -------
    CircuitBreakerRegistry
        Global registry.
    """
    return _GlobalRegistry.get()


def configure_resilience(config: ResilienceConfig) -> CircuitBreakerRegistry:
    """Configure global resilience settings.

    Parameters
    ----------
    config
        Resilience configuration.

    Returns
    -------
    CircuitBreakerRegistry
        Configured registry.
    """
    return _GlobalRegistry.configure(config)


# =============================================================================
# Middleware Protocol and Implementation
# =============================================================================


class ResilienceMiddlewareProtocol(ABC):
    """Protocol for resilience middleware.

    Define the interface for middleware that adds resilience
    to operation execution. Uses ExecutionContext for rich context.
    """

    @abstractmethod
    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute before operation invocation.

        Parameters
        ----------
        ctx
            Execution context with operation_id, params, metadata.

        Returns
        -------
        dict[str, Any]
            Context data for after_invoke.
        """
        ...

    @abstractmethod
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Execute after successful operation invocation.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Possibly modified result.
        """
        ...

    @abstractmethod
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Execute on operation error.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception | None
            Exception to raise, or None to suppress.
        """
        ...


class ResilienceMiddleware(Middleware):
    """Middleware that adds retry and circuit breaker behavior.

    This middleware integrates with operation executors to provide
    automatic resilience for operations.

    Parameters
    ----------
    config
        Resilience configuration.
    breaker_registry
        Circuit breaker registry.
    on_retry
        Callback for retry events.
    """

    def __init__(
        self,
        config: ResilienceConfig | None = None,
        breaker_registry: CircuitBreakerRegistry | None = None,
        on_retry: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        """Initialize middleware."""
        self._config = config or ResilienceConfig()
        self._breakers = breaker_registry or CircuitBreakerRegistry(self._config)
        self._on_retry = on_retry

    @property
    def config(self) -> ResilienceConfig:
        """Get configuration.

        Returns
        -------
        ResilienceConfig
            Current configuration.
        """
        return self._config

    @property
    def breaker_registry(self) -> CircuitBreakerRegistry:
        """Get circuit breaker registry.

        Returns
        -------
        CircuitBreakerRegistry
            The registry.
        """
        return self._breakers

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Check circuit breaker before invocation.

        Parameters
        ----------
        ctx
            Execution context with operation_id and params.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.
        """
        category = _get_category(ctx.operation_id)
        if self._config.circuit_breaker_enabled and category:
            breaker = self._breakers.get_breaker(category)
            breaker.allow_request()

        return {
            "start_time": time.monotonic(),
            "category": category,
        }

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Record success for circuit breaker.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        _ = ctx  # Unused but required by interface

        category = mw_context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_success()

        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Record failure for circuit breaker.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception
            The original exception.
        """
        _ = ctx  # Unused but required by interface

        category = mw_context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_failure()

        return exc


def _get_category(op_id: str) -> str | None:
    """Extract category from operation ID.

    Parameters
    ----------
    op_id
        Operation identifier.

    Returns
    -------
    str | None
        Category or None.
    """
    if "." in op_id:
        return op_id.split(".", maxsplit=1)[0]
    return None


# =============================================================================
# Utility Functions
# =============================================================================


def create_service_unavailable_error(
    operation_id: str,
    retry_after: float,
) -> ProblemDetail:
    """Create a service unavailable error for circuit breaker.

    Parameters
    ----------
    operation_id
        Operation that was blocked.
    retry_after
        Seconds until retry is possible.

    Returns
    -------
    ProblemDetail
        Error detail.
    """
    return ProblemDetail(
        type="urn:codeintel:cli:error/service-unavailable",
        title="Service Unavailable",
        detail=f"Operation {operation_id} is unavailable due to repeated failures",
        status=503,
        extensions={"retry_after": retry_after},
    )


__all__ = [
    "DEFAULT_NETWORK_POLICY",
    "DEFAULT_STORAGE_POLICY",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitBreakerStatus",
    "CircuitOpenError",
    "CircuitState",
    "ResilienceConfig",
    "ResilienceMiddleware",
    "ResilienceMiddlewareProtocol",
    "RetryCallback",
    "RetryContext",
    "RetryOptions",
    "RetryPolicy",
    "RetryableError",
    "configure_resilience",
    "create_service_unavailable_error",
    "execute_cli_with_retry",
    "execute_cli_with_retry_async",
    "execute_with_retry",
    "execute_with_retry_async",
    "get_breaker_registry",
    "with_retry",
]
