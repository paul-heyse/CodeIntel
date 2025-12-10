"""Compatibility shim for observability module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.observability`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.observability import ObservabilityConfig

    # New (preferred):
    from codeintel.cli.observability import ObservabilityConfig
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.observability' (module) is deprecated. "
    "The observability package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.observability._observability import (
    ObservabilityConfig,
    ObservabilityMiddleware,
    OperationMetrics,
    StructuredLogFormatter,
    configure_structured_logging,
    get_observability_middleware,
    get_operation_metrics,
)

__all__ = [
    "ObservabilityConfig",
    "ObservabilityMiddleware",
    "OperationMetrics",
    "StructuredLogFormatter",
    "configure_structured_logging",
    "get_observability_middleware",
    "get_operation_metrics",
]
