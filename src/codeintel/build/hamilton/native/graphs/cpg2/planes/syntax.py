"""Syntax-plane CPG node and edge assembly."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.graphs.assembly import ensure_table_columns, rename_table_columns
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
    lookup_keys,
)
from codeintel.build.hamilton.native.graphs.filter_helpers import plan_filter_or_fallback
from codeintel.build.tabular.arrow_ops import iter_array_values, normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import scalar_from_compute
from codeintel.build.tabular.compute_masks import is_valid_expr, is_valid_mask
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.extras_ops import extras_kv_from_payload
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.plan_ops import HashJoinSpec, materialize_plan
from codeintel.core.columnar.arrowdsl import join_safe_projection
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.rows import empty_table_for_table

SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"


@dataclass(frozen=True)
class SyntaxEdgeDiagnostics:
    """Diagnostics for syntax edge resolution."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class SyntaxNodeDiagnostics:
    """Diagnostics for syntax node resolution."""

    total_nodes: int
    resolved_nodes: int


def _syntax_anchor_map(syntax_nodes: pa.Table, *, include_source_pk_json: bool = True) -> pa.Table:
    """Return anchor map for syntax nodes.

    Returns
    -------
    pyarrow.Table
        Anchor map containing syntax node identifiers.
    """
    normalized = canonicalize_for_table(syntax_nodes, table_key=SYNTAX_NODES_TABLE_KEY)
    normalized = join_safe_projection(normalize_table_for_join(normalized))
    anchors = build_anchor_map(
        normalized,
        table_key=SYNTAX_NODES_TABLE_KEY,
        pk_columns=identity_keys(SYNTAX_NODES_TABLE_KEY),
        include_source_pk_json=include_source_pk_json,
    )
    return join_safe_projection(normalize_table_for_join(anchors))


def cpg2_nodes__syntax_nodes(
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG syntax nodes from core.syntax_nodes.

    Returns
    -------
    pyarrow.Table
        CPG node table for syntax nodes.
    """
    if syntax_nodes.num_rows == 0:
        return _empty_node_table()
    required = set(identity_keys(SYNTAX_NODES_TABLE_KEY))
    if not required.issubset(set(syntax_nodes.column_names)):
        return _empty_node_table()
    base = ensure_table_columns(
        syntax_nodes,
        (
            "repo",
            "commit",
            "rel_path",
            "producer",
            "node_id",
            "start_byte",
            "end_byte",
            "extras",
        ),
    )
    normalized = join_safe_projection(normalize_table_for_join(base))
    if "extras_kv" not in normalized.column_names:
        extras_kv = _extras_kv_column(normalized, column_name="extras")
        normalized = normalized.append_column("extras_kv", extras_kv)
    join_keys = ["repo", "commit", "rel_path", "producer", "node_id"]
    precheck = finalize_join_keys(
        normalized,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        precheck,
        table_key=SYNTAX_NODES_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    normalized = precheck.good
    anchor_map = join_safe_projection(
        normalize_table_for_join(_syntax_anchor_map(normalized, include_source_pk_json=True))
    )
    left_plan = build_table_plan(
        table=normalized,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "producer": E.cast(E.field("producer"), "string"),
                "node_id": E.cast(E.field("node_id"), "string"),
                "start_byte": E.field("start_byte"),
                "end_byte": E.field("end_byte"),
                "extras_kv": E.field("extras_kv"),
            }
        ),
    )
    right_plan = build_table_plan(
        table=anchor_map,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "producer": E.cast(E.field("producer"), "string"),
                "node_id": E.cast(E.field("node_id"), "string"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            }
        ),
    )
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            right_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            how="left outer",
            left_output=[
                "repo",
                "commit",
                "rel_path",
                "producer",
                "node_id",
                "start_byte",
                "end_byte",
                "extras_kv",
            ],
            right_output=["cpg_node_id", "source_pk_json"],
        ),
    )
    joined = joined.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("rel_path", "ascending"),
            ("producer", "ascending"),
            ("node_id", "ascending"),
        ]
    )
    joined = materialize_plan(joined, use_threads=True)
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "SYNTAX_NODE",
            "source_table_key": SYNTAX_NODES_TABLE_KEY,
            "extras": None,
            "extras_kv": None,
        },
    )
    selected = ensure_table_columns(joined, _CPG_NODE_COLUMNS)
    if diagnostics is not None:
        resolved = _count_valid(selected, "cpg_node_id")
        diagnostics["syntax_nodes"] = SyntaxNodeDiagnostics(
            total_nodes=selected.num_rows,
            resolved_nodes=resolved,
        )
    return selected


def cpg2_edges__syntax_edges(
    syntax_edges: pa.Table,
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG syntax edges from core.syntax_edges.

    Returns
    -------
    pyarrow.Table
        CPG edge table for syntax edges.
    """
    if syntax_edges.num_rows == 0:
        return _empty_edge_table()
    join_keys = lookup_keys(SYNTAX_NODES_TABLE_KEY, "full")
    required = set(join_keys) | {"parent_node_id", "child_node_id"}
    if not required.issubset(set(syntax_edges.column_names)):
        return _empty_edge_table()
    anchor_map = join_safe_projection(
        normalize_table_for_join(_syntax_anchor_map(syntax_nodes, include_source_pk_json=False))
    )
    normalized_edges = join_safe_projection(
        normalize_table_for_join(
            canonicalize_for_table(syntax_edges, table_key="core.syntax_edges")
        )
    )
    join_keys = ["repo", "commit", "rel_path", "producer", "parent_node_id", "child_node_id"]
    precheck = finalize_join_keys(
        normalized_edges,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        precheck,
        table_key="core.syntax_edges",
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    normalized_edges = precheck.good
    anchor_plan = build_table_plan(
        table=anchor_map,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "producer": E.cast(E.field("producer"), "string"),
                "node_id": E.cast(E.field("node_id"), "string"),
                "cpg_node_id": E.field("cpg_node_id"),
            }
        ),
    )
    parent_plan = build_table_plan(
        table=normalized_edges,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "rel_path": E.cast(E.field("rel_path"), "string"),
                "producer": E.cast(E.field("producer"), "string"),
                "node_id": E.cast(E.field("parent_node_id"), "string"),
                "child_node_id": E.cast(E.field("child_node_id"), "string"),
                "child_ordinal": E.field("child_ordinal"),
            }
        ),
    )
    parent_join = parent_plan.hash_join(
        right=anchor_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            right_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            how="left outer",
            left_output=[
                "repo",
                "commit",
                "rel_path",
                "producer",
                "child_node_id",
                "child_ordinal",
            ],
            right_output=["cpg_node_id"],
        ),
    )
    parent_join = parent_join.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("rel_path", "ascending"),
            ("producer", "ascending"),
            ("child_node_id", "ascending"),
            ("child_ordinal", "ascending"),
        ]
    )
    parent_join = materialize_plan(parent_join, use_threads=True)
    if parent_join.num_rows == 0:
        return _empty_edge_table()
    child_plan = build_table_plan(
        table=parent_join,
        options=TablePlanOptions(
            projection={
                "repo": E.field("repo"),
                "commit": E.field("commit"),
                "rel_path": E.field("rel_path"),
                "producer": E.field("producer"),
                "node_id": E.cast(E.field("child_node_id"), "string"),
                "child_ordinal": E.field("child_ordinal"),
                "src_cpg_node_id": E.field("cpg_node_id"),
            }
        ),
    )
    child_join = child_plan.hash_join(
        right=anchor_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            right_keys=["repo", "commit", "rel_path", "producer", "node_id"],
            how="left outer",
            left_output=[
                "repo",
                "commit",
                "rel_path",
                "producer",
                "child_ordinal",
                "src_cpg_node_id",
            ],
            right_output=["cpg_node_id"],
            output_suffix_for_right="_child",
        ),
    )
    child_join = child_join.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("rel_path", "ascending"),
            ("producer", "ascending"),
            ("src_cpg_node_id", "ascending"),
            ("cpg_node_id_child", "ascending"),
            ("child_ordinal", "ascending"),
        ]
    )
    child_join = materialize_plan(child_join, use_threads=True)
    if child_join.num_rows == 0:
        return _empty_edge_table()
    child_join = rename_table_columns(
        child_join,
        {"cpg_node_id_child": "dst_cpg_node_id", "child_ordinal": "ordinal"},
    )
    child_join = append_constant_columns(
        child_join,
        {
            "edge_kind": "AST",
            "edge_layer": "SYNTAX",
            "extras": None,
            "extras_kv": None,
        },
    )
    selected = ensure_table_columns(child_join, _CPG_EDGE_COLUMNS)

    expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
    filtered = plan_filter_or_fallback(selected, expr)
    if diagnostics is not None:
        resolved = filtered.num_rows
        diagnostics["syntax_edges"] = SyntaxEdgeDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=resolved,
            dropped_edges=selected.num_rows - resolved,
        )
    return filtered


def _count_valid(table: pa.Table, column: str) -> int:
    if column not in table.column_names:
        return 0
    total = scalar_from_compute("sum", [is_valid_mask(table[column])])
    if isinstance(total, (int, float)):
        return int(total)
    return 0


def _empty_node_table() -> pa.Table:
    return empty_table_for_table(CPG_NODES_TABLE_KEY)


def _empty_edge_table() -> pa.Table:
    return empty_table_for_table(CPG_EDGES_TABLE_KEY)


_CPG_NODE_COLUMNS = (
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
)

_CPG_EDGE_COLUMNS = (
    "repo",
    "commit",
    "src_cpg_node_id",
    "dst_cpg_node_id",
    "edge_kind",
    "edge_layer",
    "rel_path",
    "ordinal",
    "extras",
    "extras_kv",
)


def _extras_kv_column(table: pa.Table, *, column_name: str) -> pa.Array:
    map_type = pa.map_(pa.string(), pa.string())
    if column_name not in table.column_names:
        return pa.nulls(table.num_rows, type=map_type)
    values = [
        extras_kv_from_payload(value) for value in iter_array_values(table.column(column_name))
    ]
    return pa.array(values, type=map_type)


__all__ = [
    "SyntaxEdgeDiagnostics",
    "SyntaxNodeDiagnostics",
    "cpg2_edges__syntax_edges",
    "cpg2_nodes__syntax_nodes",
]
