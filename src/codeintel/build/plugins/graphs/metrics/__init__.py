"""Graph metrics plugins.

This package contains plugins for computing graph metrics:

- CoreMetricsPlugin: Compute core graph metrics (centrality, etc.)
"""

from __future__ import annotations

from codeintel.build.plugins.graphs.metrics.core import CoreMetricsPlugin

__all__ = [
    "CoreMetricsPlugin",
]
