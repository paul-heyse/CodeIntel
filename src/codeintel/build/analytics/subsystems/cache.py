"""Subsystem cache row builders for profile caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from codeintel.core.schemas.generated_rows import columns_for_table_key
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"


def build_subsystem_profile_cache_frame(
    snapshot: SnapshotRef,
    subsystems_frame: pl.LazyFrame,
    subsystem_graph_metrics_frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build cache rows for analytics.subsystem_profile_cache as a LazyFrame.

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
    pl.LazyFrame
        LazyFrame containing subsystem profile cache rows.
    """
    subsystems = _filter_frame_by_snapshot(subsystems_frame, snapshot)
    metrics = _filter_frame_by_snapshot(subsystem_graph_metrics_frame, snapshot)
    joined = subsystems.join(
        metrics,
        on=["repo", "commit", "subsystem_id"],
        how="left",
    )
    columns = _profile_cache_columns()
    return _ensure_columns(joined, columns)


def build_subsystem_profile_cache_rows(
    snapshot: SnapshotRef,
    subsystems_frame: pl.LazyFrame,
    subsystem_graph_metrics_frame: pl.LazyFrame,
) -> list[SubsystemProfileCacheRow]:
    """Build cache rows for analytics.subsystem_profile_cache.

    Returns
    -------
    list[SubsystemProfileCacheRow]
        Cache rows for subsystem profiles.
    """
    frame = build_subsystem_profile_cache_frame(
        snapshot,
        subsystems_frame=subsystems_frame,
        subsystem_graph_metrics_frame=subsystem_graph_metrics_frame,
    )
    rows = frame.collect().to_dicts()
    return [cast("SubsystemProfileCacheRow", row) for row in rows]


def _filter_frame_by_snapshot(frame: pl.LazyFrame, snapshot: SnapshotRef) -> pl.LazyFrame:
    available = set(frame.columns)
    if "repo" in available:
        frame = frame.filter(pl.col("repo") == snapshot.repo)
    if "commit" in available:
        frame = frame.filter(pl.col("commit") == snapshot.commit)
    return frame


def _profile_cache_columns() -> tuple[str, ...]:
    columns = columns_for_table_key(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY)
    if not columns:
        msg = f"No schema columns registered for {SUBSYSTEM_PROFILE_CACHE_TABLE_KEY}"
        raise ValueError(msg)
    return tuple(columns)


def _ensure_columns(frame: pl.LazyFrame, columns: tuple[str, ...]) -> pl.LazyFrame:
    existing = set(frame.columns)
    missing = [name for name in columns if name not in existing]
    for name in missing:
        frame = frame.with_columns(pl.lit(None).alias(name))
    return frame.select(list(columns))


__all__ = [
    "build_subsystem_profile_cache_frame",
    "build_subsystem_profile_cache_rows",
]
