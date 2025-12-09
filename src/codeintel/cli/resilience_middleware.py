"""Resilience middleware for operation execution.

.. deprecated:: 1.0
    This module is deprecated. Use :mod:`codeintel.cli.resilience` instead.

This module re-exports from the unified resilience module for
backward compatibility.
"""

from __future__ import annotations

import warnings

from codeintel.cli.resilience import (
    CircuitBreakerRegistry,
    CircuitBreakerStatus,
    ResilienceConfig,
    ResilienceMiddleware,
    configure_resilience,
    execute_cli_with_retry,
    get_breaker_registry,
)

warnings.warn(
    "codeintel.cli.resilience_middleware is deprecated, use codeintel.cli.resilience instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export execute_cli_with_retry as execute_with_retry for backward compat
execute_with_retry = execute_cli_with_retry


def is_retryable(exc: Exception, policy: object) -> bool:
    """Check if exception is retryable.

    .. deprecated:: 1.0
        Use ``policy.is_retryable(exc)`` instead.

    Parameters
    ----------
    exc
        Exception to check.
    policy
        Retry policy with ``is_retryable`` method.

    Returns
    -------
    bool
        True if retryable.
    """
    is_retryable_method = getattr(policy, "is_retryable", None)
    if callable(is_retryable_method):
        result = is_retryable_method(exc)
        return bool(result)
    retryable_exceptions = getattr(policy, "retryable_exceptions", None)
    if retryable_exceptions is not None:
        return isinstance(exc, retryable_exceptions)
    return False


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
