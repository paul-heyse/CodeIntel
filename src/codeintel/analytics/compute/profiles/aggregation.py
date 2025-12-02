"""Pure computation for profile aggregation.

This module provides functions to aggregate function-level metrics
into profile summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ProfileAggregates:
    """Aggregated metrics for a profile.

    Attributes
    ----------
    total_functions
        Total number of functions.
    total_loc
        Total lines of code.
    avg_complexity
        Average cyclomatic complexity.
    avg_typedness
        Average type annotation coverage.
    typed_count
        Number of fully typed functions.
    partial_typed_count
        Number of partially typed functions.
    untyped_count
        Number of untyped functions.
    """

    total_functions: int = 0
    total_loc: int = 0
    avg_complexity: float = 0.0
    avg_typedness: float = 0.0
    typed_count: int = 0
    partial_typed_count: int = 0
    untyped_count: int = 0
    complexity_buckets: dict[str, int] = field(default_factory=dict)


@dataclass
class FunctionMetricInput:
    """Input data for profile aggregation.

    Attributes
    ----------
    loc
        Lines of code.
    complexity
        Cyclomatic complexity.
    typedness_ratio
        Type annotation coverage ratio.
    typedness_bucket
        Typedness classification.
    complexity_bucket
        Complexity classification.
    """

    loc: int
    complexity: int
    typedness_ratio: float
    typedness_bucket: str
    complexity_bucket: str


def aggregate_function_metrics(
    metrics: Sequence[FunctionMetricInput],
) -> ProfileAggregates:
    """Aggregate function metrics into profile summary.

    Parameters
    ----------
    metrics
        Sequence of function metrics to aggregate.

    Returns
    -------
    ProfileAggregates
        Aggregated profile metrics.

    Examples
    --------
    >>> metrics = [
    ...     FunctionMetricInput(
    ...         loc=10,
    ...         complexity=2,
    ...         typedness_ratio=1.0,
    ...         typedness_bucket="typed",
    ...         complexity_bucket="low",
    ...     ),
    ...     FunctionMetricInput(
    ...         loc=20,
    ...         complexity=5,
    ...         typedness_ratio=0.5,
    ...         typedness_bucket="partial",
    ...         complexity_bucket="medium",
    ...     ),
    ... ]
    >>> agg = aggregate_function_metrics(metrics)
    >>> agg.total_functions
    2
    >>> agg.total_loc
    30
    """
    if not metrics:
        return ProfileAggregates()

    total_functions = len(metrics)
    total_loc = sum(m.loc for m in metrics)
    total_complexity = sum(m.complexity for m in metrics)
    total_typedness = sum(m.typedness_ratio for m in metrics)

    typed_count = sum(1 for m in metrics if m.typedness_bucket == "typed")
    partial_count = sum(1 for m in metrics if m.typedness_bucket == "partial")
    untyped_count = sum(1 for m in metrics if m.typedness_bucket == "untyped")

    complexity_buckets: dict[str, int] = {}
    for m in metrics:
        bucket = m.complexity_bucket
        complexity_buckets[bucket] = complexity_buckets.get(bucket, 0) + 1

    return ProfileAggregates(
        total_functions=total_functions,
        total_loc=total_loc,
        avg_complexity=total_complexity / total_functions if total_functions else 0.0,
        avg_typedness=total_typedness / total_functions if total_functions else 0.0,
        typed_count=typed_count,
        partial_typed_count=partial_count,
        untyped_count=untyped_count,
        complexity_buckets=complexity_buckets,
    )


def compute_profile_stats(aggregates: ProfileAggregates) -> dict[str, float]:
    """Compute derived statistics from profile aggregates.

    Parameters
    ----------
    aggregates
        Aggregated profile metrics.

    Returns
    -------
    dict[str, float]
        Dictionary of computed statistics.
    """
    total = aggregates.total_functions or 1

    return {
        "typed_ratio": aggregates.typed_count / total,
        "partial_ratio": aggregates.partial_typed_count / total,
        "untyped_ratio": aggregates.untyped_count / total,
        "avg_loc": aggregates.total_loc / total,
        "avg_complexity": aggregates.avg_complexity,
        "avg_typedness": aggregates.avg_typedness,
    }


__all__ = [
    "FunctionMetricInput",
    "ProfileAggregates",
    "aggregate_function_metrics",
    "compute_profile_stats",
]
