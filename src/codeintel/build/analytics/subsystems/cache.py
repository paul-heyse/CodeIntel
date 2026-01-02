"""Subsystem cache row builders for profile caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

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


__all__ = [
    "build_subsystem_profile_cache_rows",
]
