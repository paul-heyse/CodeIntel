"""Resilience middleware for operation execution.

Integrate retry policies and circuit breakers into the OperationExecutor
pipeline, providing automatic resilience for operations.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from codeintel.cli.cli_middleware import OperationMiddleware
from codeintel.cli.cli_resilience import (
    CircuitBreaker,
    RetryPolicy,
)
from codeintel.cli.error_taxonomy import INTERNAL_ERROR, StructuredCliError
from codeintel.cli.results import CliResult

LOG = logging.getLogger(__name__)


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


class CircuitBreakerRegistry:
    """Registry of circuit breakers by operation category.

    Maintain separate circuit breakers for different categories
    so failures in one category don't affect others.

    Parameters
    ----------
    config
        Resilience configuration.
    """

    def __init__(self, config: ResilienceConfig) -> None:
        """Initialize registry."""
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}

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
                failure_count=self._get_failure_count(breaker),
                last_failure_time=self._get_last_failure_time(breaker),
            )
            for key, breaker in self._breakers.items()
        }

    @staticmethod
    def _get_failure_count(breaker: CircuitBreaker) -> int:
        """Get failure count from breaker.

        Parameters
        ----------
        breaker
            Circuit breaker.

        Returns
        -------
        int
            Failure count.
        """
        # Access internal state - this is the registry's job to expose status
        return getattr(breaker, "_failure_count", 0)

    @staticmethod
    def _get_last_failure_time(breaker: CircuitBreaker) -> float:
        """Get last failure time from breaker.

        Parameters
        ----------
        breaker
            Circuit breaker.

        Returns
        -------
        float
            Last failure timestamp.
        """
        # Access internal state - this is the registry's job to expose status
        return getattr(breaker, "_last_failure_time", 0.0)

    def reset(self, key: str | None = None) -> None:
        """Reset circuit breaker(s).

        Parameters
        ----------
        key
            Specific key to reset, or None for all.
        """
        if key is not None:
            if key in self._breakers:
                del self._breakers[key]
        else:
            self._breakers.clear()


class ResilienceMiddleware(OperationMiddleware):
    """Middleware that adds retry and circuit breaker behavior.

    This middleware integrates with the OperationExecutor to provide
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

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Check circuit breaker before invocation.

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
        _ = params  # Unused but required by interface

        # Get circuit breaker for operation category
        category = _get_category(op_id)
        if self._config.circuit_breaker_enabled and category:
            breaker = self._breakers.get_breaker(category)
            # allow_request raises CircuitOpenError if open
            breaker.allow_request()

        return {
            "start_time": time.monotonic(),
            "category": category,
        }

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record success for circuit breaker.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        _ = op_id, result  # Unused but required by interface

        category = context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_success()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record failure for circuit breaker.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        _ = op_id, exc  # Unused but required by interface

        category = context.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_failure()


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


def execute_with_retry[T](
    handler: Callable[..., CliResult[T]],
    params: dict[str, Any],
    policy: RetryPolicy,
    *,
    operation_id: str = "",
    on_retry: Callable[[str, int, Exception], None] | None = None,
) -> CliResult[T]:
    """Execute handler with retry policy.

    Parameters
    ----------
    handler
        Handler function.
    params
        Handler parameters.
    policy
        Retry policy.
    operation_id
        Operation identifier for logging.
    on_retry
        Callback for retry events.

    Returns
    -------
    CliResult[T]
        Handler result.

    Raises
    ------
    StructuredCliError
        If retry loop completes without result.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            result = handler(**params)
        except policy.retryable_exceptions as e:
            last_exception = e

            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                LOG.warning(
                    "Operation %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    operation_id,
                    attempt + 1,
                    policy.max_attempts,
                    delay,
                    e,
                )
                if on_retry:
                    on_retry(operation_id, attempt + 1, e)
                time.sleep(delay)
        else:
            # Success or non-retryable result
            return result

    # All retries exhausted
    if last_exception:
        raise last_exception

    # Should not reach here
    raise StructuredCliError(
        error_code=INTERNAL_ERROR,
        detail="Retry loop completed without result",
    )


def is_retryable(exc: Exception, policy: RetryPolicy) -> bool:
    """Check if exception is retryable.

    Parameters
    ----------
    exc
        Exception to check.
    policy
        Retry policy.

    Returns
    -------
    bool
        True if retryable.
    """
    return isinstance(exc, policy.retryable_exceptions)


# Global registry for circuit breakers
_BREAKER_REGISTRY: CircuitBreakerRegistry | None = None


def get_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns
    -------
    CircuitBreakerRegistry
        Global registry.
    """
    global _BREAKER_REGISTRY  # noqa: PLW0603
    if _BREAKER_REGISTRY is None:
        _BREAKER_REGISTRY = CircuitBreakerRegistry(ResilienceConfig())
    return _BREAKER_REGISTRY


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
    global _BREAKER_REGISTRY  # noqa: PLW0603
    _BREAKER_REGISTRY = CircuitBreakerRegistry(config)
    return _BREAKER_REGISTRY


__all__ = [
    "CircuitBreakerRegistry",
    "CircuitBreakerStatus",
    "ResilienceConfig",
    "ResilienceMiddleware",
    "configure_resilience",
    "execute_with_retry",
    "get_breaker_registry",
    "is_retryable",
]
