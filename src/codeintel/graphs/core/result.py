"""Graph plugin result types.

This module re-exports unified plugin result types from codeintel.core.plugins,
providing backward compatibility for existing graph plugin code.

The canonical definitions now live in codeintel.core.plugins.result.
"""

from __future__ import annotations

from codeintel.core.plugins.result import (
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)

# Backward-compatible aliases for graph-specific naming
GraphPluginResult = PluginResult
"""Alias for PluginResult for backward compatibility."""

GraphPluginStatus = PluginStatus
"""Alias for PluginStatus for backward compatibility."""

GraphPluginRunRecord = PluginExecutionRecord
"""Alias for PluginExecutionRecord for backward compatibility."""

__all__ = [
    "GraphPluginResult",
    "GraphPluginRunRecord",
    "GraphPluginStatus",
    # Also export canonical names for migration
    "PluginExecutionRecord",
    "PluginResult",
    "PluginStatus",
]
