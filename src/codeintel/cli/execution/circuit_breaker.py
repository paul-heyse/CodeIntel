"""Circuit breaker pattern for failure protection.

Provide circuit breakers that prevent repeated failures from cascading
and allow systems to recover gracefully.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


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
class CircuitBreakerRegistry:
    """Registry of circuit breakers by operation category.

    Maintain separate circuit breakers for different categories
    so failures in one category don't affect others.

    Parameters
    ----------
    failure_threshold
        Failures before circuit opens.
    recovery_timeout
        Seconds before attempting recovery.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0

    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict, init=False)

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
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
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


class _GlobalRegistry:
    """Singleton manager for the global circuit breaker registry."""

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
    def configure(
        cls,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> CircuitBreakerRegistry:
        """Configure the global registry.

        Parameters
        ----------
        failure_threshold
            Failures before circuit opens.
        recovery_timeout
            Seconds before attempting recovery.

        Returns
        -------
        CircuitBreakerRegistry
            Configured registry.
        """
        cls._instance = CircuitBreakerRegistry(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
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


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitBreakerStatus",
    "CircuitOpenError",
    "CircuitState",
    "get_breaker_registry",
]
