"""Resilience patterns for CLI operations.

This package provides:

- ``RetryPolicy``: Configurable retry with exponential backoff
- ``CircuitBreaker``: Circuit breaker pattern for failure protection
- ``ResilienceMiddleware``: Middleware for automatic resilience
"""

from __future__ import annotations

# Re-export all from the implementation module
from codeintel.cli.resilience._resilience import (
    DEFAULT_NETWORK_POLICY,
    DEFAULT_STORAGE_POLICY,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerStatus,
    CircuitOpenError,
    CircuitState,
    ResilienceConfig,
    ResilienceMiddleware,
    ResilienceMiddlewareProtocol,
    RetryableError,
    RetryCallback,
    RetryContext,
    RetryOptions,
    RetryPolicy,
    configure_resilience,
    create_service_unavailable_error,
    execute_cli_with_retry,
    execute_cli_with_retry_async,
    execute_with_retry,
    execute_with_retry_async,
    get_breaker_registry,
    with_retry,
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
