"""Pure computation for profile feature extraction.

This module provides functions to extract features from profiles
for classification and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.compute.profiles.aggregation import ProfileAggregates


@dataclass(frozen=True)
class ProfileFeatures:
    """Extracted features for a profile.

    Attributes
    ----------
    size_category
        Size classification (small, medium, large).
    complexity_category
        Complexity classification (simple, moderate, complex).
    typedness_category
        Type coverage classification (typed, partial, untyped).
    quality_score
        Overall quality score (0.0 to 1.0).
    """

    size_category: str
    complexity_category: str
    typedness_category: str
    quality_score: float


def extract_profile_features(aggregates: ProfileAggregates) -> ProfileFeatures:
    """Extract classification features from profile aggregates.

    Parameters
    ----------
    aggregates
        Aggregated profile metrics.

    Returns
    -------
    ProfileFeatures
        Extracted feature set.

    Examples
    --------
    >>> from codeintel.analytics.compute.profiles.aggregation import ProfileAggregates
    >>> agg = ProfileAggregates(
    ...     total_functions=10,
    ...     total_loc=100,
    ...     avg_complexity=3.0,
    ...     avg_typedness=0.8,
    ...     typed_count=8,
    ...     partial_typed_count=1,
    ...     untyped_count=1,
    ... )
    >>> features = extract_profile_features(agg)
    >>> features.typedness_category
    'typed'
    """
    # Size classification
    if aggregates.total_loc < 100:
        size_category = "small"
    elif aggregates.total_loc < 1000:
        size_category = "medium"
    else:
        size_category = "large"

    # Complexity classification
    if aggregates.avg_complexity < 5:
        complexity_category = "simple"
    elif aggregates.avg_complexity < 10:
        complexity_category = "moderate"
    else:
        complexity_category = "complex"

    # Typedness classification
    if aggregates.avg_typedness >= 0.8:
        typedness_category = "typed"
    elif aggregates.avg_typedness >= 0.3:
        typedness_category = "partial"
    else:
        typedness_category = "untyped"

    # Quality score (weighted combination)
    typedness_weight = 0.4
    complexity_weight = 0.3
    size_weight = 0.3

    typedness_score = aggregates.avg_typedness
    complexity_score = max(0, 1 - (aggregates.avg_complexity / 20))
    size_score = 1.0 if aggregates.total_functions > 0 else 0.0

    quality_score = (
        typedness_weight * typedness_score
        + complexity_weight * complexity_score
        + size_weight * size_score
    )

    return ProfileFeatures(
        size_category=size_category,
        complexity_category=complexity_category,
        typedness_category=typedness_category,
        quality_score=min(1.0, max(0.0, quality_score)),
    )


__all__ = [
    "ProfileFeatures",
    "extract_profile_features",
]

