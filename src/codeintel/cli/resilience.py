"""Compatibility shim for resilience module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.resilience`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.resilience import RetryPolicy, CircuitBreaker

    # New (preferred):
    from codeintel.cli.resilience import RetryPolicy, CircuitBreaker
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.resilience' (module) is deprecated. "
    "The resilience package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
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
