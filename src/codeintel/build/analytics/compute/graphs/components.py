"""Re-export component helpers from build.graphs.compute.metrics."""

from __future__ import annotations

from codeintel.build.graphs.compute.metrics.components import (
    component_ids_undirected,
    component_metadata,
    global_graph_stats,
)

__all__ = [
    "component_ids_undirected",
    "component_metadata",
    "global_graph_stats",
]
