"""Subsystem cache row builders for profile and coverage caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemCoverageCacheRow as SubsystemCoverageCacheRow,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

PROFILE_CACHE_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "modules_json",
    "entrypoints_json",
    "internal_edge_count",
    "external_edge_count",
    "fan_in",
    "fan_out",
    "function_count",
    "avg_risk_score",
    "max_risk_score",
    "high_risk_function_count",
    "risk_level",
    "import_in_degree",
    "import_out_degree",
    "import_pagerank",
    "import_betweenness",
    "import_closeness",
    "import_layer",
    "created_at",
)

COVERAGE_CACHE_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "function_count",
    "risk_level",
    "avg_risk_score",
    "max_risk_score",
    "test_count",
    "passed_test_count",
    "failed_test_count",
    "skipped_test_count",
    "xfail_test_count",
    "flaky_test_count",
    "total_functions_covered",
    "avg_functions_covered",
    "max_functions_covered",
    "min_functions_covered",
    "function_coverage_ratio",
    "created_at",
)


def build_subsystem_profile_cache_rows(
    snapshot: SnapshotRef,
    subsystems_frame: pl.LazyFrame,
    subsystem_graph_metrics_frame: pl.LazyFrame,
) -> list[SubsystemProfileCacheRow]:
    """Build cache rows for analytics.subsystem_profile_cache.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    subsystems_frame
        Subsystems dataset frame.
    subsystem_graph_metrics_frame
        Subsystem graph metrics dataset frame.

    Returns
    -------
    list[SubsystemProfileCacheRow]
        Cache rows for subsystem profiles.
    """
    subsystems = _filter_frame_by_snapshot(subsystems_frame, snapshot)
    metrics = _filter_frame_by_snapshot(subsystem_graph_metrics_frame, snapshot)
    joined = subsystems.join(
        metrics,
        on=["repo", "commit", "subsystem_id"],
        how="left",
    )
    frame = _ensure_columns(joined, PROFILE_CACHE_COLUMNS)
    rows = frame.collect().to_dicts()
    return [cast("SubsystemProfileCacheRow", row) for row in rows]


def build_subsystem_coverage_cache_rows(
    snapshot: SnapshotRef,
    subsystems_frame: pl.LazyFrame,
    test_profile_frame: pl.LazyFrame,
) -> list[SubsystemCoverageCacheRow]:
    """Build cache rows for analytics.subsystem_coverage_cache.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    subsystems_frame
        Subsystems dataset frame.
    test_profile_frame
        Test profile dataset frame.

    Returns
    -------
    list[SubsystemCoverageCacheRow]
        Cache rows for subsystem coverage.
    """
    subsystems = _filter_frame_by_snapshot(subsystems_frame, snapshot)
    coverage_stats = _coverage_stats(test_profile_frame, snapshot)
    joined = subsystems.join(
        coverage_stats,
        on=["repo", "commit", "subsystem_id"],
        how="left",
    ).with_columns(
        _coverage_ratio_expr().alias("function_coverage_ratio"),
    )
    frame = _ensure_columns(joined, COVERAGE_CACHE_COLUMNS)
    rows = frame.collect().to_dicts()
    return [cast("SubsystemCoverageCacheRow", row) for row in rows]


def _filter_frame_by_snapshot(frame: pl.LazyFrame, snapshot: SnapshotRef) -> pl.LazyFrame:
    available = set(frame.columns)
    if "repo" in available:
        frame = frame.filter(pl.col("repo") == snapshot.repo)
    if "commit" in available:
        frame = frame.filter(pl.col("commit") == snapshot.commit)
    return frame


def _ensure_columns(frame: pl.LazyFrame, columns: tuple[str, ...]) -> pl.LazyFrame:
    existing = set(frame.columns)
    missing = [name for name in columns if name not in existing]
    for name in missing:
        frame = frame.with_columns(pl.lit(None).alias(name))
    return frame.select(list(columns))


def _coverage_stats(
    test_profile_frame: pl.LazyFrame,
    snapshot: SnapshotRef,
) -> pl.LazyFrame:
    frame = _filter_frame_by_snapshot(test_profile_frame, snapshot)
    if "primary_subsystem_id" not in set(frame.columns):
        return pl.LazyFrame(schema={"repo": pl.Utf8, "commit": pl.Utf8, "subsystem_id": pl.Utf8})
    frame = frame.filter(pl.col("primary_subsystem_id").is_not_null())
    status = pl.col("status")
    covered_count = pl.col("functions_covered_count").fill_null(0)
    grouped = frame.group_by(["repo", "commit", "primary_subsystem_id"]).agg(
        [
            pl.len().alias("test_count"),
            _status_count_expr(status, "passed").alias("passed_test_count"),
            _status_count_expr(status, "failed").alias("failed_test_count"),
            _status_count_expr(status, "skipped").alias("skipped_test_count"),
            _status_count_expr(status, "xfail").alias("xfail_test_count"),
            pl.sum(pl.col("flaky").fill_null(value=False).cast(pl.Int64)).alias("flaky_test_count"),
            covered_count.sum().alias("total_functions_covered"),
            covered_count.mean().alias("avg_functions_covered"),
            covered_count.max().alias("max_functions_covered"),
            covered_count.min().alias("min_functions_covered"),
        ]
    )
    return grouped.rename({"primary_subsystem_id": "subsystem_id"})


def _status_count_expr(status: pl.Expr, value: str) -> pl.Expr:
    return pl.sum(pl.when(status == value).then(1).otherwise(0))


def _coverage_ratio_expr() -> pl.Expr:
    return (
        pl.when(pl.col("function_count").is_null() | (pl.col("function_count") == 0))
        .then(None)
        .otherwise(pl.col("total_functions_covered") / pl.col("function_count"))
    )


__all__ = [
    "build_subsystem_coverage_cache_rows",
    "build_subsystem_profile_cache_rows",
]
