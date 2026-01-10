"""Subsystem cache row builders for profile caches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows, normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.schemas.row_models import columns_for_table_key

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


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
    join_keys = ("repo", "commit", "subsystem_id")
    subsystems = _prepare_join_frame(
        subsystems_frame,
        snapshot=snapshot,
        join_keys=join_keys,
        table_key="analytics.subsystems",
    )
    metrics = _prepare_join_frame(
        subsystem_graph_metrics_frame,
        snapshot=snapshot,
        join_keys=join_keys,
        table_key="analytics.subsystem_graph_metrics",
    )
    joined = _join_subsystem_metrics(subsystems, metrics, join_keys=join_keys)
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


def _prepare_join_frame(
    frame: pa.Table,
    *,
    snapshot: SnapshotRef,
    join_keys: tuple[str, ...],
    table_key: str,
) -> pa.Table:
    scoped = _filter_table_by_snapshot(frame, snapshot)
    precheck = finalize_join_keys(
        scoped,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        precheck,
        table_key=table_key,
        target_name=None,
        join_keys=join_keys,
    )
    return normalize_table_for_join(precheck.good)


def _projection_with_key_cast(
    columns: Sequence[str],
    *,
    join_keys: tuple[str, ...],
) -> dict[str, Expression]:
    projection = {name: E.field(name) for name in columns}
    for key in join_keys:
        if key in projection:
            projection[key] = E.cast(E.field(key), "string")
    return projection


def _right_output_columns(
    right_columns: Sequence[str],
    *,
    join_keys: tuple[str, ...],
    left_columns: Sequence[str],
) -> list[str]:
    return [name for name in right_columns if name not in join_keys and name not in left_columns]


def _plan_to_table(plan: Plan) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(_INTERNAL_PLAN_TABLE_KEY, mode="tolerant"),
    )
    return result.good


def _join_subsystem_metrics(
    subsystems: pa.Table,
    metrics: pa.Table,
    *,
    join_keys: tuple[str, ...],
) -> pa.Table:
    left_columns = list(subsystems.column_names)
    right_columns = list(metrics.column_names)
    left_project = _projection_with_key_cast(left_columns, join_keys=join_keys)
    right_project = _projection_with_key_cast(right_columns, join_keys=join_keys)
    right_output = _right_output_columns(
        right_columns,
        join_keys=join_keys,
        left_columns=left_columns,
    )
    left_plan = build_table_plan(
        table=subsystems,
        options=TablePlanOptions(projection=left_project),
    )
    right_plan = build_table_plan(
        table=metrics,
        options=TablePlanOptions(projection=right_project),
    )
    joined_plan = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            how="left outer",
            left_output=list(left_project.keys()),
            right_output=right_output,
        ),
    )
    return _plan_to_table(joined_plan)


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
