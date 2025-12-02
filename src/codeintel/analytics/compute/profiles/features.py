"""Pure computation for profile feature extraction.

This module provides functions to extract features from profiles
for classification and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from codeintel.analytics.compute.profiles.aggregation import ProfileAggregates

# Size thresholds (lines of code)
SMALL_MODULE_THRESHOLD: Final[int] = 100
LARGE_MODULE_THRESHOLD: Final[int] = 1000

# Complexity thresholds (average cyclomatic complexity)
LOW_COMPLEXITY_THRESHOLD: Final[float] = 5.0
HIGH_COMPLEXITY_THRESHOLD: Final[float] = 10.0

# Typedness thresholds (annotation ratio)
HIGH_TYPED_RATIO: Final[float] = 0.8
LOW_TYPED_RATIO: Final[float] = 0.3

# Quality score weights
TYPEDNESS_WEIGHT: Final[float] = 0.4
COMPLEXITY_WEIGHT: Final[float] = 0.3
SIZE_WEIGHT: Final[float] = 0.3

# Complexity normalization factor
COMPLEXITY_NORMALIZATION: Final[float] = 20.0


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
    if aggregates.total_loc < SMALL_MODULE_THRESHOLD:
        size_category = "small"
    elif aggregates.total_loc < LARGE_MODULE_THRESHOLD:
        size_category = "medium"
    else:
        size_category = "large"

    # Complexity classification
    if aggregates.avg_complexity < LOW_COMPLEXITY_THRESHOLD:
        complexity_category = "simple"
    elif aggregates.avg_complexity < HIGH_COMPLEXITY_THRESHOLD:
        complexity_category = "moderate"
    else:
        complexity_category = "complex"

    # Typedness classification
    if aggregates.avg_typedness >= HIGH_TYPED_RATIO:
        typedness_category = "typed"
    elif aggregates.avg_typedness >= LOW_TYPED_RATIO:
        typedness_category = "partial"
    else:
        typedness_category = "untyped"

    # Quality score (weighted combination)
    typedness_score = aggregates.avg_typedness
    complexity_score = max(0.0, 1 - (aggregates.avg_complexity / COMPLEXITY_NORMALIZATION))
    size_score = 1.0 if aggregates.total_functions > 0 else 0.0

    quality_score = (
        TYPEDNESS_WEIGHT * typedness_score
        + COMPLEXITY_WEIGHT * complexity_score
        + SIZE_WEIGHT * size_score
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
