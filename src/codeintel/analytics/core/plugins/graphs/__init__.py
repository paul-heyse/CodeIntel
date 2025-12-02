"""Graph analytics plugins using the new protocol.

This module provides graph-level analytics plugins migrated
to the new unified plugin protocol.
"""

from __future__ import annotations

from codeintel.analytics.core.plugins.graphs.core_metrics import (
    CoreGraphMetricsPlugin,
)

__all__ = [
    "CoreGraphMetricsPlugin",
]
