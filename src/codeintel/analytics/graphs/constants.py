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

__all__ = [
    "CENTRALITY_SAMPLE_LIMIT",
    "EIGEN_MAX_ITER",
    "RICH_CLUB_PERCENTILE",
]
