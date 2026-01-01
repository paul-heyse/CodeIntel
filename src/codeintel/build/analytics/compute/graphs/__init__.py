"""Graph metric primitives for analytics orchestration.

This package provides primitive graph computation helpers used by analytics
plugins. It wraps pure computation functions from codeintel.build.graphs.compute
and provides analytics-specific data structures and orchestration.

Submodules
----------
types
    Data classes for metric results.
conversions
    ID conversion and normalization utilities.
centrality
    Centrality computation for directed and undirected graphs.
components
    Component analysis and global graph statistics.
projections
    Bipartite graph projection and metrics.
structural
    Structural graph metrics (clustering, cores, etc.).
cfg
    Control flow graph metrics.
dfg
    Data flow graph metrics.
"""

from __future__ import annotations

from codeintel.build.analytics.compute.graphs.centrality import (
    centrality_directed,
    centrality_undirected,
    neighbor_stats,
)
from codeintel.build.analytics.compute.graphs.cfg import (
    build_cfg_graph,
    cfg_avg_shortest_path_length,
    cfg_centralities,
    cfg_dominance_metrics,
    cfg_longest_path_length,
    cfg_reachable_nodes,
)
from codeintel.build.analytics.compute.graphs.components import (
    component_ids_undirected,
    component_metadata,
    global_graph_stats,
)
from codeintel.build.analytics.compute.graphs.conversions import (
    log_empty_graph,
    log_projection_skipped,
    safe_float,
)
from codeintel.build.analytics.compute.graphs.dfg import (
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
    dfg_path_lengths,
)
from codeintel.build.analytics.compute.graphs.projections import (
    bipartite_degrees,
    build_projection_graph,
    community_ids,
    projection_metrics,
)
from codeintel.build.analytics.compute.graphs.structural import (
    bounded_simple_path_count,
    structural_metrics,
)
from codeintel.build.analytics.compute.graphs.types import (
    BipartiteDegrees,
    CentralityBundle,
    ComponentBundle,
    DominanceMetrics,
    GlobalGraphStats,
    NeighborStats,
    ProjectionMetrics,
    StructuralMetrics,
)

__all__ = [
    "BipartiteDegrees",
    "CentralityBundle",
    "ComponentBundle",
    "DominanceMetrics",
    "GlobalGraphStats",
    "NeighborStats",
    "ProjectionMetrics",
    "StructuralMetrics",
    "bipartite_degrees",
    "bounded_simple_path_count",
    "build_cfg_graph",
    "build_dfg_graph",
    "build_projection_graph",
    "centrality_directed",
    "centrality_undirected",
    "cfg_avg_shortest_path_length",
    "cfg_centralities",
    "cfg_dominance_metrics",
    "cfg_longest_path_length",
    "cfg_reachable_nodes",
    "community_ids",
    "component_ids_undirected",
    "component_metadata",
    "dfg_centralities",
    "dfg_component_stats",
    "dfg_path_lengths",
    "global_graph_stats",
    "log_empty_graph",
    "log_projection_skipped",
    "neighbor_stats",
    "projection_metrics",
    "safe_float",
    "structural_metrics",
]
