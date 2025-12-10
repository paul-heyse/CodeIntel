"""Compatibility shim for telemetry module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.observability`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.telemetry import TelemetryConfig, TelemetryProvider

    # New (preferred):
    from codeintel.cli.observability import TelemetryConfig, TelemetryProvider
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.telemetry' is deprecated. "
    "Use 'codeintel.cli.observability' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.observability._telemetry import (
    TelemetryConfig,
    TelemetryProvider,
    TracingMiddleware,
    get_telemetry_provider,
)

__all__ = [
    "TelemetryConfig",
    "TelemetryProvider",
    "TracingMiddleware",
    "get_telemetry_provider",
]
