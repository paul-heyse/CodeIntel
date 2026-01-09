"""GOID plane CPG nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.filter_helpers import plan_filter_or_fallback
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_masks import is_valid_expr
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan, join_safe_projection
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.primitives import resolve_join_safe_columns

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
GOIDS_TABLE_KEY = "core.goids"


@dataclass(frozen=True)
class GoidNodeDiagnostics:
    """Diagnostics for GOID CPG node emission."""

    total_rows: int
    resolved_rows: int
    dropped_rows: int


def cpg2_nodes__goids(
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes from GOID rows.

    Returns
    -------
    pyarrow.Table
        CPG node table for GOIDs.
    """
    required = {"goid_h128", "repo", "commit", "rel_path"}
    if not required.issubset(set(goids.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized = canonicalize_for_table(goids, table_key=GOIDS_TABLE_KEY)
    allowlist = _join_safe_allowlist(GOIDS_TABLE_KEY)
    normalized = join_safe_projection(
        normalize_table_for_join(normalized, allowed_columns=allowlist),
        allowed_columns=allowlist,
    )
    join_keys = ["goid_h128"]
    left_precheck = finalize_join_keys(
        normalized,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        left_precheck,
        table_key=GOIDS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    normalized = left_precheck.good
    anchors = build_anchor_map(
        normalized,
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = join_safe_projection(
        normalize_table_for_join(anchors, allowed_columns=allowlist),
        allowed_columns=allowlist,
    )
    right_precheck = finalize_join_keys(
        anchors,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        right_precheck,
        table_key=GOIDS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    anchors = right_precheck.good
    left_plan = build_table_plan(
        table=normalized,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
            },
            filter_expr=E.is_valid("goid_h128"),
        ),
    )
    right_plan = build_table_plan(
        table=anchors,
        options=TablePlanOptions(
            projection={
                "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            },
            filter_expr=E.is_valid("goid_h128"),
        ),
    )
    joined_plan = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["goid_h128"],
            right_keys=["goid_h128"],
            how="left outer",
            left_output=["repo", "commit", "rel_path", "goid_h128"],
            right_output=["cpg_node_id", "source_pk_json"],
        ),
    )
    joined = _plan_to_table(joined_plan, use_threads=True)
    if joined.num_rows != 0:
        joined = joined.take(
            stable_sort_indices(
                joined,
                sort_keys=[
                    ("repo", "ascending"),
                    ("commit", "ascending"),
                    ("goid_h128", "ascending"),
                ],
            )
        )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "GOID",
            "source_table_key": GOIDS_TABLE_KEY,
            "start_byte": None,
            "end_byte": None,
            "extras": None,
            "extras_kv": None,
        },
    )
    selected = joined.select(
        [
            "repo",
            "commit",
            "cpg_node_id",
            "node_kind",
            "source_table_key",
            "source_pk_json",
            "rel_path",
            "start_byte",
            "end_byte",
            "extras",
            "extras_kv",
        ]
    )
    filtered = _filter_valid_nodes(selected)
    if diagnostics is not None:
        diagnostics["goids"] = GoidNodeDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    if "cpg_node_id" not in table.column_names:
        return table
    return plan_filter_or_fallback(table, is_valid_expr("cpg_node_id"))


def _join_safe_allowlist(table_key: str | None) -> tuple[str, ...]:
    if table_key is None:
        return ()
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return ()
    return resolve_join_safe_columns(schema)


def _plan_to_table(plan: Plan, *, use_threads: bool) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    if not use_threads:
        execution_ctx = replace(execution_ctx, use_threads=False)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


__all__ = ["GoidNodeDiagnostics", "cpg2_nodes__goids"]
