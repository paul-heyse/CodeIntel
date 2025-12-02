"""Pure computation functions for profile analytics.

This module provides side-effect-free functions for:
- Aggregating function metrics into profiles
- Computing profile features and statistics
"""

from __future__ import annotations

from codeintel.analytics.compute.profiles.aggregation import (
    ProfileAggregates,
    aggregate_function_metrics,
    compute_profile_stats,
)
from codeintel.analytics.compute.profiles.features import (
    ProfileFeatures,
    extract_profile_features,
)

__all__ = [
    "ProfileAggregates",
    "ProfileFeatures",
    "aggregate_function_metrics",
    "compute_profile_stats",
    "extract_profile_features",
]
