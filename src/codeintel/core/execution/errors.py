"""Centralized error definitions for plugin execution.

This module re-exports error types from the unified core/errors module
for backward compatibility.

.. deprecated:: 5.0.0
    Import from ``codeintel.core.errors`` instead.
"""

from __future__ import annotations

import warnings

from codeintel.core.errors.execution import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginSkipRequestError,
    PluginTimeoutError,
)

# Emit deprecation warning on import of this module
warnings.warn(
    "codeintel.core.execution.errors is deprecated. "
    "Import from codeintel.core.errors or codeintel.core.errors.execution instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "PluginFatalError",
    "PluginSkipRequestError",
    "PluginSkippedError",
    "PluginTimeoutError",
]
