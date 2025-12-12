"""Test profile aggregation and feature extraction computation.

Test the pure computation functions for aggregating function metrics
and extracting profile features for classification.
"""

from __future__ import annotations

from codeintel.analytics.compute.profiles.aggregation import (
    FunctionMetricInput,
    ProfileAggregates,
    aggregate_function_metrics,
    compute_profile_stats,
)
from codeintel.analytics.compute.profiles.features import (
    COMPLEXITY_NORMALIZATION,
    COMPLEXITY_WEIGHT,
    HIGH_COMPLEXITY_THRESHOLD,
    HIGH_TYPED_RATIO,
    LARGE_MODULE_THRESHOLD,
    LOW_COMPLEXITY_THRESHOLD,
    LOW_TYPED_RATIO,
    SIZE_WEIGHT,
    SMALL_MODULE_THRESHOLD,
    TYPEDNESS_WEIGHT,
    ProfileFeatures,
    extract_profile_features,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_true,
)

EXPECTED_FUNCTIONS_0 = 0
EXPECTED_FUNCTIONS_1 = 1
EXPECTED_FUNCTIONS_3 = 3
EXPECTED_FUNCTIONS_4 = 4
EXPECTED_FUNCTIONS_5 = 5
EXPECTED_FUNCTIONS_10 = 10
EXPECTED_LOC_0 = 0
EXPECTED_LOC_20 = 20
EXPECTED_LOC_60 = 60
EXPECTED_LOC_100 = 100
EXPECTED_TYPED_COUNT_2 = 2
EXPECTED_TYPED_COUNT_6 = 6
EXPECTED_TYPED_COUNT_7 = 7
EXPECTED_PARTIAL_COUNT_1 = 1
EXPECTED_PARTIAL_COUNT_3 = 3
EXPECTED_UNTYPED_COUNT_1 = 1
EXPECTED_COMPLEXITY_3 = 3.0
EXPECTED_AVG_COMPLEXITY = 4.0
EXPECTED_AVG_LOC_25 = 25.0
EXPECTED_QUALITY_SCORE_0_85 = 0.85
EXPECTED_QUALITY_HIGH_COMPLEX = 0.7
QUALITY_TOLERANCE = 0.01
RATIO_TOLERANCE = 0.001
WEIGHT_SUM = 1.0
TYPED_RATIO_0_5 = 0.5
PARTIAL_RATIO_0_3 = 0.3
UNTYPED_RATIO_0_2 = 0.2


def _make_typed_function(
    loc: int = 10,
    complexity: int = 2,
) -> FunctionMetricInput:
    """
    Create a fully typed function metric.

    Parameters
    ----------
    loc
        Lines of code for the function.
    complexity
        Cyclomatic complexity.

    Returns
    -------
    FunctionMetricInput
        A typed function metric input.
    """
    return FunctionMetricInput(
        loc=loc,
        complexity=complexity,
        typedness_ratio=1.0,
        typedness_bucket="typed",
        complexity_bucket="low",
    )


def _make_partial_function(
    loc: int = 20,
    complexity: int = 5,
) -> FunctionMetricInput:
    """
    Create a partially typed function metric.

    Parameters
    ----------
    loc
        Lines of code for the function.
    complexity
        Cyclomatic complexity.

    Returns
    -------
    FunctionMetricInput
        A partial function metric input.
    """
    return FunctionMetricInput(
        loc=loc,
        complexity=complexity,
        typedness_ratio=0.5,
        typedness_bucket="partial",
        complexity_bucket="medium",
    )


def _make_untyped_function(
    loc: int = 30,
    complexity: int = 8,
) -> FunctionMetricInput:
    """
    Create an untyped function metric.

    Parameters
    ----------
    loc
        Lines of code for the function.
    complexity
        Cyclomatic complexity.

    Returns
    -------
    FunctionMetricInput
        An untyped function metric input.
    """
    return FunctionMetricInput(
        loc=loc,
        complexity=complexity,
        typedness_ratio=0.0,
        typedness_bucket="untyped",
        complexity_bucket="high",
    )


def _make_realistic_codebase() -> list[FunctionMetricInput]:
    """
    Create a realistic distribution of function metrics.

    Simulates a typical codebase:
    - 60% typed, 25% partial, 15% untyped
    - Mix of simple and complex functions
    - Various sizes

    Returns
    -------
    list[FunctionMetricInput]
        A list of realistic function metrics.
    """
    base_loc = 10
    loc_step = 5
    base_complexity = 2
    complexity_mod = 3
    partial_base_loc = 15
    partial_loc_step = 10
    partial_base_complexity = 4
    untyped_loc = 50
    untyped_complexity = 10

    metrics: list[FunctionMetricInput] = [
        _make_typed_function(
            loc=base_loc + i * loc_step, complexity=base_complexity + i % complexity_mod
        )
        for i in range(6)
    ]
    metrics.extend(
        _make_partial_function(
            loc=partial_base_loc + i * partial_loc_step, complexity=partial_base_complexity + i
        )
        for i in range(3)
    )
    metrics.append(_make_untyped_function(loc=untyped_loc, complexity=untyped_complexity))
    return metrics


def test_aggregates_default_values() -> None:
    """Verify default values for ProfileAggregates."""
    agg = ProfileAggregates()
    expect_equal(agg.total_functions, EXPECTED_FUNCTIONS_0)
    expect_equal(agg.total_loc, EXPECTED_LOC_0)
    expect_equal(agg.avg_complexity, 0.0)
    expect_equal(agg.complexity_buckets, {})


def test_aggregates_custom_values() -> None:
    """Create ProfileAggregates with custom values."""
    agg = ProfileAggregates(
        total_functions=EXPECTED_FUNCTIONS_10,
        total_loc=500,
        avg_complexity=5.5,
        avg_typedness=0.8,
        typed_count=EXPECTED_TYPED_COUNT_7,
        partial_typed_count=2,
        untyped_count=1,
        complexity_buckets={"low": 5, "medium": 3, "high": 2},
    )
    expect_equal(agg.total_functions, EXPECTED_FUNCTIONS_10)
    expect_equal(agg.typed_count, EXPECTED_TYPED_COUNT_7)


def test_metric_create() -> None:
    """Create function metric input."""
    metric = FunctionMetricInput(
        loc=25,
        complexity=3,
        typedness_ratio=0.75,
        typedness_bucket="partial",
        complexity_bucket="low",
    )
    expected_loc = 25
    expected_ratio = 0.75
    expect_equal(metric.loc, expected_loc)
    expect_equal(metric.typedness_ratio, expected_ratio)


def test_aggregate_empty_metrics() -> None:
    """Empty metrics return default aggregates."""
    result = aggregate_function_metrics([])
    expect_equal(result.total_functions, EXPECTED_FUNCTIONS_0)


def test_aggregate_single_function() -> None:
    """Aggregate single function metrics."""
    metric = _make_typed_function(loc=20, complexity=3)
    result = aggregate_function_metrics([metric])
    expect_equal(result.total_functions, EXPECTED_FUNCTIONS_1)
    expect_equal(result.total_loc, EXPECTED_LOC_20)
    expect_equal(result.avg_complexity, EXPECTED_COMPLEXITY_3)
    expect_equal(result.typed_count, EXPECTED_FUNCTIONS_1)


def test_aggregate_multiple_functions() -> None:
    """Aggregate multiple function metrics."""
    metrics = [
        _make_typed_function(loc=10, complexity=2),
        _make_typed_function(loc=20, complexity=4),
        _make_partial_function(loc=30, complexity=6),
    ]
    result = aggregate_function_metrics(metrics)
    expect_equal(result.total_functions, EXPECTED_FUNCTIONS_3)
    expect_equal(result.total_loc, EXPECTED_LOC_60)
    expect_true(abs(result.avg_complexity - EXPECTED_AVG_COMPLEXITY) < RATIO_TOLERANCE)


def test_aggregate_typedness_counts() -> None:
    """Count functions by typedness bucket."""
    metrics = [
        _make_typed_function(),
        _make_typed_function(),
        _make_partial_function(),
        _make_untyped_function(),
    ]
    result = aggregate_function_metrics(metrics)
    expect_equal(result.typed_count, EXPECTED_TYPED_COUNT_2)
    expect_equal(result.partial_typed_count, EXPECTED_PARTIAL_COUNT_1)
    expect_equal(result.untyped_count, EXPECTED_UNTYPED_COUNT_1)


def test_aggregate_complexity_buckets() -> None:
    """Count functions by complexity bucket."""
    metrics = [
        FunctionMetricInput(10, 2, 1.0, "typed", "low"),
        FunctionMetricInput(20, 5, 1.0, "typed", "low"),
        FunctionMetricInput(30, 8, 0.5, "partial", "medium"),
        FunctionMetricInput(40, 15, 0.0, "untyped", "high"),
    ]
    result = aggregate_function_metrics(metrics)
    expect_equal(result.complexity_buckets.get("low"), EXPECTED_TYPED_COUNT_2)
    expect_equal(result.complexity_buckets.get("medium"), EXPECTED_PARTIAL_COUNT_1)
    expect_equal(result.complexity_buckets.get("high"), EXPECTED_UNTYPED_COUNT_1)


def test_aggregate_average_typedness() -> None:
    """Compute average typedness ratio."""
    metrics = [
        FunctionMetricInput(10, 2, 1.0, "typed", "low"),
        FunctionMetricInput(20, 3, 0.5, "partial", "low"),
        FunctionMetricInput(30, 4, 0.0, "untyped", "low"),
    ]
    result = aggregate_function_metrics(metrics)
    expected = (1.0 + 0.5 + 0.0) / 3
    expect_true(abs(result.avg_typedness - expected) < RATIO_TOLERANCE)


def test_aggregate_realistic_codebase() -> None:
    """Aggregate realistic codebase metrics."""
    metrics = _make_realistic_codebase()
    result = aggregate_function_metrics(metrics)
    expect_equal(result.total_functions, EXPECTED_FUNCTIONS_10)
    expect_equal(result.typed_count, EXPECTED_TYPED_COUNT_6)
    expect_equal(result.partial_typed_count, EXPECTED_PARTIAL_COUNT_3)
    expect_equal(result.untyped_count, EXPECTED_UNTYPED_COUNT_1)


def test_stats_empty_aggregates() -> None:
    """Handle empty aggregates without division by zero."""
    agg = ProfileAggregates()
    stats = compute_profile_stats(agg)
    expect_equal(stats["typed_ratio"], 0.0)


def test_stats_all_typed() -> None:
    """Compute stats for fully typed codebase."""
    agg = ProfileAggregates(
        total_functions=10,
        total_loc=100,
        avg_complexity=3.0,
        avg_typedness=1.0,
        typed_count=10,
        partial_typed_count=0,
        untyped_count=0,
    )
    stats = compute_profile_stats(agg)
    expect_equal(stats["typed_ratio"], 1.0)
    expect_equal(stats["partial_ratio"], 0.0)


def test_stats_mixed_typedness() -> None:
    """Compute stats for mixed typedness."""
    agg = ProfileAggregates(
        total_functions=10,
        total_loc=200,
        avg_complexity=5.0,
        avg_typedness=0.6,
        typed_count=5,
        partial_typed_count=3,
        untyped_count=2,
    )
    stats = compute_profile_stats(agg)
    expect_equal(stats["typed_ratio"], TYPED_RATIO_0_5)
    expect_equal(stats["partial_ratio"], PARTIAL_RATIO_0_3)
    expect_equal(stats["untyped_ratio"], UNTYPED_RATIO_0_2)


def test_stats_avg_loc() -> None:
    """Compute average LOC per function."""
    agg = ProfileAggregates(total_functions=4, total_loc=EXPECTED_LOC_100)
    stats = compute_profile_stats(agg)
    expect_equal(stats["avg_loc"], EXPECTED_AVG_LOC_25)


def test_stats_all_present() -> None:
    """Verify all expected stats are returned."""
    agg = ProfileAggregates(
        total_functions=5,
        total_loc=50,
        avg_complexity=4.0,
        avg_typedness=0.7,
        typed_count=3,
        partial_typed_count=1,
        untyped_count=1,
    )
    stats = compute_profile_stats(agg)
    expected_keys = {
        "typed_ratio",
        "partial_ratio",
        "untyped_ratio",
        "avg_loc",
        "avg_complexity",
        "avg_typedness",
    }
    expect_equal(set(stats.keys()), expected_keys)


def test_features_create() -> None:
    """Create profile features."""
    features = ProfileFeatures(
        size_category="medium",
        complexity_category="moderate",
        typedness_category="typed",
        quality_score=EXPECTED_QUALITY_SCORE_0_85,
    )
    expect_equal(features.size_category, "medium")
    expect_equal(features.quality_score, EXPECTED_QUALITY_SCORE_0_85)


def test_features_is_frozen() -> None:
    """Features dataclass is immutable."""
    features = ProfileFeatures(
        size_category="small",
        complexity_category="simple",
        typedness_category="typed",
        quality_score=1.0,
    )
    assert_frozen(features, "quality_score", 0.5)


def test_extract_small_simple_typed() -> None:
    """Extract features for small, simple, fully typed module."""
    agg = ProfileAggregates(
        total_functions=5,
        total_loc=50,
        avg_complexity=2.0,
        avg_typedness=0.95,
        typed_count=5,
    )
    features = extract_profile_features(agg)
    expect_equal(features.size_category, "small")
    expect_equal(features.complexity_category, "simple")
    expect_equal(features.typedness_category, "typed")


def test_extract_large_complex_untyped() -> None:
    """Extract features for large, complex, untyped module."""
    agg = ProfileAggregates(
        total_functions=50,
        total_loc=2000,
        avg_complexity=12.0,
        avg_typedness=0.1,
        untyped_count=45,
    )
    features = extract_profile_features(agg)
    expect_equal(features.size_category, "large")
    expect_equal(features.complexity_category, "complex")
    expect_equal(features.typedness_category, "untyped")


def test_extract_medium_moderate_partial() -> None:
    """Extract features for medium-sized, moderate, partially typed module."""
    agg = ProfileAggregates(
        total_functions=20,
        total_loc=500,
        avg_complexity=7.0,
        avg_typedness=0.5,
        partial_typed_count=10,
    )
    features = extract_profile_features(agg)
    expect_equal(features.size_category, "medium")
    expect_equal(features.complexity_category, "moderate")
    expect_equal(features.typedness_category, "partial")


def test_quality_score_bounds() -> None:
    """Quality score is bounded between 0.0 and 1.0."""
    high_agg = ProfileAggregates(
        total_functions=10,
        total_loc=100,
        avg_complexity=1.0,
        avg_typedness=1.0,
        typed_count=10,
    )
    high_features = extract_profile_features(high_agg)
    expect_true(0.0 <= high_features.quality_score <= 1.0)

    low_agg = ProfileAggregates(
        total_functions=0,
        total_loc=0,
        avg_complexity=25.0,
        avg_typedness=0.0,
    )
    low_features = extract_profile_features(low_agg)
    expect_true(0.0 <= low_features.quality_score <= 1.0)


def test_quality_score_weights() -> None:
    """Quality score uses correct weights."""
    total_weight = TYPEDNESS_WEIGHT + COMPLEXITY_WEIGHT + SIZE_WEIGHT
    expect_true(abs(total_weight - WEIGHT_SUM) < RATIO_TOLERANCE)


def test_threshold_constants() -> None:
    """Verify threshold constants are reasonable."""
    expect_true(SMALL_MODULE_THRESHOLD < LARGE_MODULE_THRESHOLD)
    expect_true(LOW_COMPLEXITY_THRESHOLD < HIGH_COMPLEXITY_THRESHOLD)
    expect_true(LOW_TYPED_RATIO < HIGH_TYPED_RATIO)


def test_size_boundary_small_medium() -> None:
    """Test size classification at small/medium boundary."""
    small = ProfileAggregates(
        total_functions=5,
        total_loc=SMALL_MODULE_THRESHOLD - 1,
    )
    small_features = extract_profile_features(small)
    expect_equal(small_features.size_category, "small")

    medium = ProfileAggregates(
        total_functions=5,
        total_loc=SMALL_MODULE_THRESHOLD,
    )
    medium_features = extract_profile_features(medium)
    expect_equal(medium_features.size_category, "medium")


def test_size_boundary_medium_large() -> None:
    """Test size classification at medium/large boundary."""
    medium = ProfileAggregates(
        total_functions=20,
        total_loc=LARGE_MODULE_THRESHOLD - 1,
    )
    medium_features = extract_profile_features(medium)
    expect_equal(medium_features.size_category, "medium")

    large = ProfileAggregates(
        total_functions=20,
        total_loc=LARGE_MODULE_THRESHOLD,
    )
    large_features = extract_profile_features(large)
    expect_equal(large_features.size_category, "large")


def test_complexity_boundary() -> None:
    """Test complexity classification at boundaries."""
    simple = ProfileAggregates(
        total_functions=5,
        avg_complexity=LOW_COMPLEXITY_THRESHOLD - 0.1,
    )
    simple_features = extract_profile_features(simple)
    expect_equal(simple_features.complexity_category, "simple")

    moderate = ProfileAggregates(
        total_functions=5,
        avg_complexity=LOW_COMPLEXITY_THRESHOLD,
    )
    moderate_features = extract_profile_features(moderate)
    expect_equal(moderate_features.complexity_category, "moderate")

    complex_ = ProfileAggregates(
        total_functions=5,
        avg_complexity=HIGH_COMPLEXITY_THRESHOLD,
    )
    complex_features = extract_profile_features(complex_)
    expect_equal(complex_features.complexity_category, "complex")


def test_typedness_boundary() -> None:
    """Test typedness classification at boundaries."""
    untyped = ProfileAggregates(
        total_functions=5,
        avg_typedness=LOW_TYPED_RATIO - 0.01,
    )
    untyped_features = extract_profile_features(untyped)
    expect_equal(untyped_features.typedness_category, "untyped")

    partial = ProfileAggregates(
        total_functions=5,
        avg_typedness=LOW_TYPED_RATIO,
    )
    partial_features = extract_profile_features(partial)
    expect_equal(partial_features.typedness_category, "partial")

    typed = ProfileAggregates(
        total_functions=5,
        avg_typedness=HIGH_TYPED_RATIO,
    )
    typed_features = extract_profile_features(typed)
    expect_equal(typed_features.typedness_category, "typed")


def test_complexity_normalization() -> None:
    """Quality score uses complexity normalization correctly."""
    high_complex = ProfileAggregates(
        total_functions=5,
        avg_complexity=COMPLEXITY_NORMALIZATION * 2,
        avg_typedness=1.0,
    )
    features = extract_profile_features(high_complex)

    expect_true(abs(features.quality_score - EXPECTED_QUALITY_HIGH_COMPLEX) < QUALITY_TOLERANCE)


def test_aggregate_then_extract() -> None:
    """Full pipeline from metrics to features."""
    metrics = [
        _make_typed_function(loc=20, complexity=2),
        _make_typed_function(loc=30, complexity=3),
        _make_partial_function(loc=40, complexity=4),
    ]
    aggregates = aggregate_function_metrics(metrics)
    features = extract_profile_features(aggregates)

    expect_equal(features.size_category, "small")

    expect_equal(features.complexity_category, "simple")


def test_realistic_codebase_features() -> None:
    """Extract features from realistic codebase."""
    metrics = _make_realistic_codebase()
    aggregates = aggregate_function_metrics(metrics)
    stats = compute_profile_stats(aggregates)
    features = extract_profile_features(aggregates)

    expect_equal(stats["avg_complexity"], aggregates.avg_complexity)

    min_quality = 0.3
    max_quality = 0.9
    expect_true(min_quality < features.quality_score < max_quality)
