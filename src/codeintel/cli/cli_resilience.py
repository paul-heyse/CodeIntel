"""Resilience and retry infrastructure for CLI operations.

.. deprecated:: 1.0
    This module is deprecated. Use :mod:`codeintel.cli.resilience` instead.

This module re-exports from the unified resilience module for
backward compatibility.
"""

from __future__ import annotations

import warnings

from codeintel.cli.resilience import (
    DEFAULT_NETWORK_POLICY,
    DEFAULT_STORAGE_POLICY,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RetryableError,
    RetryCallback,
    RetryContext,
    RetryPolicy,
    with_retry,
)
from codeintel.cli.resilience import (
    ResilienceMiddleware as RetryMiddleware,
)

warnings.warn(
    "codeintel.cli.cli_resilience is deprecated, use codeintel.cli.resilience instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_NETWORK_POLICY",
    "DEFAULT_STORAGE_POLICY",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "RetryCallback",
    "RetryContext",
    "RetryMiddleware",
    "RetryPolicy",
    "RetryableError",
    "with_retry",
]
