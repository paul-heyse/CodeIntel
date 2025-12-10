"""Resilience middleware and configuration.

Integrate retry and circuit breaker patterns with the execution layer
through middleware.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.execution.circuit_breaker import (
    CircuitBreakerRegistry,
    get_breaker_registry,
)
from codeintel.cli.execution.middleware import Middleware
from codeintel.cli.execution.retry import RetryPolicy

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
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
        on_retry: object | None = None,
    ) -> None:
        """Initialize middleware."""
        self._config = config or ResilienceConfig()
        self._breakers = breaker_registry or get_breaker_registry()
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
    from codeintel.cli.execution.circuit_breaker import (  # noqa: PLC0415
        _GlobalRegistry,
    )

    return _GlobalRegistry.configure(
        failure_threshold=config.circuit_failure_threshold,
        recovery_timeout=config.circuit_recovery_timeout,
    )


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
    "ResilienceConfig",
    "ResilienceMiddleware",
    "ResilienceMiddlewareProtocol",
    "configure_resilience",
    "create_service_unavailable_error",
]
