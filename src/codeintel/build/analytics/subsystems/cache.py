"""Subsystem cache row builders for profile caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows, normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.schemas.row_models import columns_for_table_key

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
    join_keys = ["repo", "commit", "subsystem_id"]
    subsystems_precheck = finalize_join_keys(
        subsystems,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        subsystems_precheck,
        table_key="analytics.subsystems",
        target_name=None,
        join_keys=join_keys,
    )
    subsystems = normalize_table_for_join(subsystems_precheck.good)
    metrics_precheck = finalize_join_keys(
        metrics,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        metrics_precheck,
        table_key="analytics.subsystem_graph_metrics",
        target_name=None,
        join_keys=join_keys,
    )
    metrics = normalize_table_for_join(metrics_precheck.good)
    left_columns = list(subsystems.column_names)
    right_columns = list(metrics.column_names)
    left_project = {name: E.field(name) for name in left_columns}
    right_project = {name: E.field(name) for name in right_columns}
    for key in join_keys:
        if key in left_project:
            left_project[key] = E.cast(E.field(key), "string")
        if key in right_project:
            right_project[key] = E.cast(E.field(key), "string")
    right_output = [
        name for name in right_columns if name not in join_keys and name not in left_columns
    ]
    joined_plan = (
        Plan.table(subsystems)
        .project(left_project)
        .hash_join(
            right=Plan.table(metrics).project(right_project),
            spec=HashJoinSpec(
                left_keys=join_keys,
                right_keys=join_keys,
                how="left outer",
                left_output=list(left_project.keys()),
                right_output=right_output,
            ),
        )
    )
    joined_plan = joined_plan.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("subsystem_id", "ascending"),
        ]
    )
    joined = materialize_plan(joined_plan, use_threads=True)
    columns = _profile_cache_columns()
    return _ensure_columns(joined, columns)


def build_subsystem_profile_cache_rows(
    snapshot: SnapshotRef,
    subsystems_frame: pa.Table,
    subsystem_graph_metrics_frame: pa.Table,
) -> list[dict[str, object]]:
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
    return list(iter_rows(frame))


def _filter_table_by_snapshot(frame: pa.Table, snapshot: SnapshotRef) -> pa.Table:
    scope = SnapshotScope.from_snapshot(snapshot)
    return scope.filter_arrow_table(frame, require_columns=True)


def _profile_cache_columns() -> tuple[str, ...]:
    columns = columns_for_table_key(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY)
    if not columns:
        msg = f"No schema columns registered for {SUBSYSTEM_PROFILE_CACHE_TABLE_KEY}"
        raise ValueError(msg)
    return tuple(columns)


def _ensure_columns(frame: pa.Table, columns: tuple[str, ...]) -> pa.Table:
    existing = set(frame.column_names)
    missing = [name for name in columns if name not in existing]
    if missing:
        constants = dict.fromkeys(missing)
        frame = append_constant_columns(frame, constants)
    return frame.select(list(columns))


__all__ = [
    "build_subsystem_profile_cache_frame",
    "build_subsystem_profile_cache_rows",
]
