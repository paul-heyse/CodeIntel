"""Shared constants for graph metrics computation.

This module centralizes tuning constants used across graph metrics modules
to ensure consistency and simplify configuration updates.
"""

from __future__ import annotations

# Centrality computation limits
CENTRALITY_SAMPLE_LIMIT: int = 500
"""Maximum nodes to sample for betweenness centrality computation."""

EIGEN_MAX_ITER: int = 200
"""Maximum iterations for eigenvector centrality convergence."""

# Structural metrics thresholds
RICH_CLUB_PERCENTILE: float = 0.1
"""Percentile threshold for rich-club coefficient calculation."""

# Symbol/config graph metrics limits
MAX_BETWEENNESS_NODES: int = 1000
"""Maximum nodes for betweenness centrality in symbol/config graphs."""

MAX_COMMUNITY_NODES: int = 5000
"""Maximum nodes for community detection in symbol graphs."""

# CFG/DFG sampling limits
MAX_CFG_CENTRALITY_SAMPLE: int = 100
"""Maximum nodes to sample for CFG betweenness centrality."""

MAX_CFG_EIGEN_SAMPLE: int = 200
"""Maximum iterations for CFG eigenvector centrality."""

MAX_DFG_CENTRALITY_SAMPLE: int = 100
"""Maximum nodes to sample for DFG betweenness centrality."""

__all__ = [
    "CENTRALITY_SAMPLE_LIMIT",
    "EIGEN_MAX_ITER",
    "MAX_BETWEENNESS_NODES",
    "MAX_CFG_CENTRALITY_SAMPLE",
    "MAX_CFG_EIGEN_SAMPLE",
    "MAX_COMMUNITY_NODES",
    "MAX_DFG_CENTRALITY_SAMPLE",
    "RICH_CLUB_PERCENTILE",
]
