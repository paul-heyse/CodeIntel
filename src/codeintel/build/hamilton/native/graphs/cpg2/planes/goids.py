"""GOID plane CPG nodes."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter_expr
from codeintel.build.tabular.compute_masks import is_valid_expr, is_valid_mask
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.columnar.rows import empty_table_for_table

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
    normalized = normalize_table_for_join(normalized)
    anchors = build_anchor_map(
        normalized,
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = normalize_table_for_join(anchors)
    left_plan = (
        Plan.table(normalized)
        .project(
            {
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
            }
        )
        .filter(E.is_valid("goid_h128"))
    )
    right_plan = (
        Plan.table(anchors)
        .project(
            {
                "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            }
        )
        .filter(E.is_valid("goid_h128"))
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
    joined = materialize_plan(joined_plan, use_threads=True)
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

    def _mask(value_table: pa.Table) -> pa.Array | pa.ChunkedArray:
        return is_valid_mask(value_table.column("cpg_node_id"))

    return safe_filter_expr(table, is_valid_expr("cpg_node_id"), fallback_mask=_mask)


__all__ = ["GoidNodeDiagnostics", "cpg2_nodes__goids"]
