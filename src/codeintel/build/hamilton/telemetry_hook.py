"""Backward compatibility alias for telemetry_hook.

This module has been moved to codeintel.build.hamilton.hooks.telemetry_hook.
All imports are re-exported for backward compatibility.

.. deprecated::
    Import from codeintel.build.hamilton.hooks instead.
"""

from __future__ import annotations

# Re-export everything from the new location for backward compatibility
from codeintel.build.hamilton.hooks.telemetry_hook import (
    NodeExecutionRecord,
    NodeTelemetryHook,
)

__all__ = [
    "NodeExecutionRecord",
    "NodeTelemetryHook",
]
