"""Graph metric plugins.

This package contains plugins that compute metrics over graph structures:

- CoreMetricsPlugin: Core function/module metrics (PageRank, centrality, components)
- SecondaryMetricsPlugin: Extended metrics (CFG, DFG, community detection)

All plugins implement the TargetPlugin protocol and are executed by the
build system via BuildExecutor.
"""

from codeintel.graphs.plugins.metrics.core import CoreMetricsPlugin
from codeintel.graphs.plugins.metrics.secondary import SecondaryMetricsPlugin

__all__ = [
    "CoreMetricsPlugin",
    "SecondaryMetricsPlugin",
]
