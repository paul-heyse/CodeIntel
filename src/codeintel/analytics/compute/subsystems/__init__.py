"""Pure computation layer for subsystem analytics.

This module provides pure functions for subsystem classification and
organization. All functions are side-effect-free and operate on in-memory
data structures.

The module re-exports core clustering and classification functions from
the subsystems domain module.
"""

from __future__ import annotations

from codeintel.analytics.subsystems.affinity import (
    build_weighted_graph,
    clusters_from_labels,
    graph_to_adjacency,
    label_propagation_nx,
    limit_clusters,
    reassign_small_clusters,
    seed_labels_from_tags,
)
from codeintel.analytics.subsystems.edge_stats import (
    compute_subsystem_edge_stats,
)
from codeintel.analytics.subsystems.risk import (
    SubsystemRisk,
    aggregate_risk,
)

__all__ = [
    "SubsystemRisk",
    "aggregate_risk",
    "build_weighted_graph",
    "clusters_from_labels",
    "compute_subsystem_edge_stats",
    "graph_to_adjacency",
    "label_propagation_nx",
    "limit_clusters",
    "reassign_small_clusters",
    "seed_labels_from_tags",
]
