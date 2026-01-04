"""Subsystem cache row builders for profile caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import ArrowJoinSpec, arrow_join_tables
from codeintel.core.schemas.generated_rows import columns_for_table_key
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"


def build_subsystem_profile_cache_frame(
    snapshot: SnapshotRef,
    subsystems_frame: pa.Table,
    subsystem_graph_metrics_frame: pa.Table,
) -> pa.Table:
    """Build cache rows for analytics.subsystem_profile_cache as an Arrow table.

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
    pa.Table
        Table containing subsystem profile cache rows.
    """
    subsystems = _filter_table_by_snapshot(subsystems_frame, snapshot)
    metrics = _filter_table_by_snapshot(subsystem_graph_metrics_frame, snapshot)
    join_spec = ArrowJoinSpec(
        on=["repo", "commit", "subsystem_id"],
        how="left",
        validate="m:1",
    )
    joined = arrow_join_tables(subsystems, metrics, spec=join_spec)
    columns = _profile_cache_columns()
    return _ensure_columns(joined, columns)


def build_subsystem_profile_cache_rows(
    snapshot: SnapshotRef,
    subsystems_frame: pa.Table,
    subsystem_graph_metrics_frame: pa.Table,
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
    rows = frame.to_pylist()
    return [cast("SubsystemProfileCacheRow", row) for row in rows]


def _filter_table_by_snapshot(frame: pa.Table, snapshot: SnapshotRef) -> pa.Table:
    available = set(frame.column_names)
    rows = frame.to_pylist()
    filtered = [
        row
        for row in rows
        if (snapshot.repo == row.get("repo") if "repo" in available else True)
        and (snapshot.commit == row.get("commit") if "commit" in available else True)
    ]
    return pa.Table.from_pylist(filtered) if filtered else pa.Table.from_pylist([])


def _profile_cache_columns() -> tuple[str, ...]:
    columns = columns_for_table_key(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY)
    if not columns:
        msg = f"No schema columns registered for {SUBSYSTEM_PROFILE_CACHE_TABLE_KEY}"
        raise ValueError(msg)
    return tuple(columns)


def _ensure_columns(frame: pa.Table, columns: tuple[str, ...]) -> pa.Table:
    existing = set(frame.column_names)
    missing = [name for name in columns if name not in existing]
    for name in missing:
        frame = frame.append_column(name, pa.array([None] * frame.num_rows))
    return frame.select(list(columns))


__all__ = [
    "build_subsystem_profile_cache_frame",
    "build_subsystem_profile_cache_rows",
]
