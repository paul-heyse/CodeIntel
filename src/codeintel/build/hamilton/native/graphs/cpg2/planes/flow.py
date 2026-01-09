"""Flow plane CPG nodes and edges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import pyarrow as pa

from codeintel.build.graphs.assembly import (
    rename_table_columns,
    select_table_columns,
)
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinals
from codeintel.build.hamilton.native.graphs.filter_helpers import plan_filter_or_fallback
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import (
    array_from_compute,
    cast_array,
)
from codeintel.build.tabular.compute_masks import is_valid_expr, is_valid_mask
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan, join_safe_projection
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.primitives import resolve_join_safe_columns

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
GOIDS_TABLE_KEY = "core.goids"


@dataclass(frozen=True)
class CfgBlockDiagnostics:
    """Diagnostics for CFG block CPG nodes."""

    total_rows: int
    resolved_rows: int
    dropped_rows: int


@dataclass(frozen=True)
class CfgEdgeDiagnostics:
    """Diagnostics for CFG edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class DfgEdgeDiagnostics:
    """Diagnostics for DFG edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class CdgEdgeDiagnostics:
    """Diagnostics for CDG edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


def cpg2_nodes__cfg_blocks(
    cfg_blocks: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes from CFG block inventory.

    Returns
    -------
    pyarrow.Table
        CPG node table for CFG blocks.
    """
    required = {"function_goid_h128", "block_idx", "file_path"}
    if not required.issubset(set(cfg_blocks.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized_blocks = canonicalize_for_table(
        cfg_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        casts={"function_goid_h128": pa.decimal128(38, 0)},
    )
    normalized_blocks = _join_ready(normalized_blocks, table_key=CFG_BLOCKS_TABLE_KEY)
    block_keys = ["function_goid_h128", "block_idx"]
    block_precheck = finalize_join_keys(
        normalized_blocks,
        required_non_null=block_keys,
        key_fields=block_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        block_precheck,
        table_key=CFG_BLOCKS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=block_keys,
    )
    normalized_blocks = block_precheck.good
    goid_ctx = _join_ready(_goid_context(goids), table_key=GOIDS_TABLE_KEY)
    goid_precheck = finalize_join_keys(
        goid_ctx,
        required_non_null=["function_goid_h128"],
        key_fields=["function_goid_h128"],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        goid_precheck,
        table_key=GOIDS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=["function_goid_h128"],
    )
    goid_ctx = goid_precheck.good
    block_plan = build_table_plan(
        table=normalized_blocks,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_idx": E.field("block_idx"),
                "file_path": E.field("file_path"),
            }
        ),
    )
    goid_plan = build_table_plan(
        table=goid_ctx,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
            }
        ),
    )
    joined = block_plan.hash_join(
        right=goid_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128"],
            right_keys=["function_goid_h128"],
            how="left outer",
            left_output=["function_goid_h128", "block_idx", "file_path"],
            right_output=["repo", "commit"],
        ),
    )
    anchors = build_anchor_map(
        normalized_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        pk_columns=identity_keys(CFG_BLOCKS_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = _join_ready(anchors, table_key=CFG_BLOCKS_TABLE_KEY)
    anchor_plan = build_table_plan(
        table=anchors,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_idx": E.field("block_idx"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            }
        ),
    )
    joined = joined.hash_join(
        right=anchor_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128", "block_idx"],
            right_keys=["function_goid_h128", "block_idx"],
            how="left outer",
            left_output=[
                "function_goid_h128",
                "block_idx",
                "file_path",
                "repo",
                "commit",
            ],
            right_output=["cpg_node_id", "source_pk_json"],
        ),
    )
    joined = joined.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("function_goid_h128", "ascending"),
            ("block_idx", "ascending"),
        ]
    )
    joined_table = _plan_to_table(joined, use_threads=True)
    joined = append_constant_columns(
        joined_table,
        {
            "node_kind": "CFG_BLOCK",
            "source_table_key": CFG_BLOCKS_TABLE_KEY,
            "start_byte": None,
            "end_byte": None,
            "extras": None,
            "extras_kv": None,
        },
    )
    joined = rename_table_columns(joined, {"file_path": "rel_path"})
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
        diagnostics["cfg_blocks"] = CfgBlockDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__cfg_edges(
    cfg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for CFG transitions.

    Returns
    -------
    pyarrow.Table
        CPG edges for CFG links.
    """
    required = {"function_goid_h128", "src_block_id", "dst_block_id"}
    if not required.issubset(set(cfg_edges.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    normalized_edges = _normalize_flow_edges(cfg_edges)
    normalized_edges = _rename_if_present(normalized_edges, {"edge_kind": "cfg_edge_kind"})
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup, table_key="graph.cfg_edges")
    joined = _join_ready(joined, table_key="graph.cfg_edges")
    joined = _join_block_anchors(joined, anchors, table_key="graph.cfg_edges")
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path", result_type=pa.string())
    joined = joined.append_column("rel_path", rel_path)
    repo = _coalesce_rel_path(joined, "src_repo", "dst_repo", result_type=pa.string())
    commit = _coalesce_rel_path(joined, "src_commit", "dst_commit", result_type=pa.string())
    joined = _upsert_column(joined, "repo", repo)
    joined = _upsert_column(joined, "commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "CFG", "edge_layer": "FLOW", "extras": None, "extras_kv": None},
    )
    joined = _assign_ordinals(
        joined,
        table_key="graph.cfg_edges",
        columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "cfg_edge_kind",
            "edge_kind",
            "edge_layer",
            "src_cpg_node_id",
            "dst_cpg_node_id",
        ],
    )
    if joined.num_rows > 0:
        sort_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("function_goid_h128", "ascending"),
            ("src_block_id", "ascending"),
            ("dst_block_id", "ascending"),
        ]
        if "cfg_edge_kind" in joined.column_names:
            sort_keys.append(("cfg_edge_kind", "ascending"))
        ordered_plan = build_table_plan(
            table=joined,
            options=TablePlanOptions(order_by=sort_keys),
        )
        joined = _plan_to_table(ordered_plan, use_threads=True)
    selected = joined.select(
        [
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
        ]
    )
    filtered = _filter_valid_edges(selected)
    if diagnostics is not None:
        diagnostics["cfg_edges"] = CfgEdgeDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__dfg_edges(
    dfg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for data-flow links.

    Returns
    -------
    pyarrow.Table
        CPG edges for DFG links.
    """
    required = {"function_goid_h128", "src_block_id", "dst_block_id"}
    if not required.issubset(set(dfg_edges.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    normalized_edges = _normalize_flow_edges(dfg_edges)
    normalized_edges = _rename_if_present(normalized_edges, {"edge_kind": "dfg_edge_kind"})
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup, table_key="graph.dfg_edges")
    joined = _join_ready(joined, table_key="graph.dfg_edges")
    joined = _join_block_anchors(joined, anchors, table_key="graph.dfg_edges")
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path", result_type=pa.string())
    joined = joined.append_column("rel_path", rel_path)
    repo = _coalesce_rel_path(joined, "src_repo", "dst_repo", result_type=pa.string())
    commit = _coalesce_rel_path(joined, "src_commit", "dst_commit", result_type=pa.string())
    joined = _upsert_column(joined, "repo", repo)
    joined = _upsert_column(joined, "commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "DFG", "edge_layer": "FLOW", "extras": None, "extras_kv": None},
    )
    joined = _assign_ordinals(
        joined,
        table_key="graph.dfg_edges",
        columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "src_var",
            "dst_var",
            "dfg_edge_kind",
            "via_phi",
            "use_kind",
            "edge_kind",
            "edge_layer",
            "src_cpg_node_id",
            "dst_cpg_node_id",
        ],
    )
    if joined.num_rows > 0:
        sort_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("function_goid_h128", "ascending"),
            ("src_block_id", "ascending"),
            ("dst_block_id", "ascending"),
        ]
        sort_keys.extend(
            [
                (name, "ascending")
                for name in ["src_var", "dst_var", "dfg_edge_kind", "via_phi", "use_kind"]
                if name in joined.column_names
            ]
        )
        ordered_plan = build_table_plan(
            table=joined,
            options=TablePlanOptions(order_by=sort_keys),
        )
        joined = _plan_to_table(ordered_plan, use_threads=True)
    selected = joined.select(
        [
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
        ]
    )
    filtered = _filter_valid_edges(selected)
    if diagnostics is not None:
        diagnostics["dfg_edges"] = DfgEdgeDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__cdg_edges(
    cdg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for control-dependence links.

    Returns
    -------
    pyarrow.Table
        CPG edges for CDG links.
    """
    required = {"function_goid_h128", "src_block_id", "dst_block_id"}
    if not required.issubset(set(cdg_edges.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    normalized_edges = _normalize_flow_edges(cdg_edges)
    normalized_edges = _rename_if_present(normalized_edges, {"edge_kind": "cdg_edge_kind"})
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup, table_key="graph.cdg_edges")
    joined = _join_ready(joined, table_key="graph.cdg_edges")
    joined = _join_block_anchors(joined, anchors, table_key="graph.cdg_edges")
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path", result_type=pa.string())
    joined = joined.append_column("rel_path", rel_path)
    repo = _coalesce_rel_path(joined, "src_repo", "dst_repo", result_type=pa.string())
    commit = _coalesce_rel_path(joined, "src_commit", "dst_commit", result_type=pa.string())
    joined = _upsert_column(joined, "repo", repo)
    joined = _upsert_column(joined, "commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "CDG", "edge_layer": "FLOW", "extras": None, "extras_kv": None},
    )
    joined = _assign_ordinals(
        joined,
        table_key="graph.cdg_edges",
        columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "via_succ_block_id",
            "cdg_edge_kind",
            "via_edge_kind",
            "edge_kind",
            "edge_layer",
            "src_cpg_node_id",
            "dst_cpg_node_id",
        ],
    )
    if joined.num_rows > 0:
        sort_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("function_goid_h128", "ascending"),
            ("src_block_id", "ascending"),
            ("dst_block_id", "ascending"),
        ]
        sort_keys.extend(
            [
                (name, "ascending")
                for name in ["via_succ_block_id", "cdg_edge_kind", "via_edge_kind"]
                if name in joined.column_names
            ]
        )
        ordered_plan = build_table_plan(
            table=joined,
            options=TablePlanOptions(order_by=sort_keys),
        )
        joined = _plan_to_table(ordered_plan, use_threads=True)
    selected = joined.select(
        [
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
        ]
    )
    filtered = _filter_valid_edges(selected)
    if diagnostics is not None:
        diagnostics["cdg_edges"] = CdgEdgeDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def _goid_context(goids: pa.Table) -> pa.Table:
    if goids.num_rows == 0:
        return pa.table({"function_goid_h128": [], "repo": [], "commit": []})
    selected = select_table_columns(goids, ["goid_h128", "repo", "commit"])
    normalized = canonicalize_for_table(
        selected,
        table_key=GOIDS_TABLE_KEY,
        casts={"goid_h128": pa.decimal128(38, 0), "repo": pa.string(), "commit": pa.string()},
    )
    return rename_table_columns(normalized, {"goid_h128": "function_goid_h128"})


def _cfg_block_lookup(cfg_blocks: pa.Table, goids: pa.Table) -> pa.Table:
    required = {"function_goid_h128", "block_id", "block_idx", "file_path"}
    if not required.issubset(set(cfg_blocks.column_names)):
        return pa.table(
            {
                "function_goid_h128": [],
                "block_id": [],
                "block_idx": [],
                "rel_path": [],
                "repo": [],
                "commit": [],
            }
        )
    normalized_blocks = canonicalize_for_table(
        select_table_columns(
            cfg_blocks, ["function_goid_h128", "block_id", "block_idx", "file_path"]
        ),
        table_key=CFG_BLOCKS_TABLE_KEY,
        casts={"function_goid_h128": pa.decimal128(38, 0), "block_id": pa.string()},
    )
    normalized_blocks = _join_ready(normalized_blocks, table_key=CFG_BLOCKS_TABLE_KEY)
    lookup_keys = ["function_goid_h128", "block_id", "block_idx"]
    lookup_precheck = finalize_join_keys(
        normalized_blocks,
        required_non_null=lookup_keys,
        key_fields=lookup_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        lookup_precheck,
        table_key=CFG_BLOCKS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=lookup_keys,
    )
    normalized_blocks = lookup_precheck.good
    goid_ctx = _join_ready(_goid_context(goids), table_key=GOIDS_TABLE_KEY)
    goid_precheck = finalize_join_keys(
        goid_ctx,
        required_non_null=["function_goid_h128"],
        key_fields=["function_goid_h128"],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        goid_precheck,
        table_key=GOIDS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=["function_goid_h128"],
    )
    goid_ctx = goid_precheck.good
    block_plan = build_table_plan(
        table=normalized_blocks,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_id": E.cast(E.field("block_id"), "string"),
                "block_idx": E.field("block_idx"),
                "file_path": E.field("file_path"),
            }
        ),
    )
    goid_plan = build_table_plan(
        table=goid_ctx,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
            }
        ),
    )
    joined = block_plan.hash_join(
        right=goid_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128"],
            right_keys=["function_goid_h128"],
            how="left outer",
            left_output=["function_goid_h128", "block_id", "block_idx", "file_path"],
            right_output=["repo", "commit"],
        ),
    )
    joined = joined.order_by(
        sort_keys=[
            ("function_goid_h128", "ascending"),
            ("block_id", "ascending"),
        ]
    )
    joined_table = _plan_to_table(joined, use_threads=True)
    joined_table = rename_table_columns(joined_table, {"file_path": "rel_path"})
    return _join_ready(
        joined_table.select(
            ["function_goid_h128", "block_id", "block_idx", "rel_path", "repo", "commit"]
        ),
        table_key=CFG_BLOCKS_TABLE_KEY,
    )


def _cfg_block_anchor(cfg_blocks: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(
        cfg_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        casts={"function_goid_h128": pa.decimal128(38, 0)},
    )
    anchors = build_anchor_map(
        normalized,
        table_key=CFG_BLOCKS_TABLE_KEY,
        pk_columns=identity_keys(CFG_BLOCKS_TABLE_KEY),
        include_source_pk_json=False,
    )
    return _join_ready(anchors, table_key=CFG_BLOCKS_TABLE_KEY)


def _normalize_flow_edges(edges: pa.Table) -> pa.Table:
    return canonicalize_for_table(
        edges,
        table_key="graph.flow_edges",
        casts={
            "function_goid_h128": pa.decimal128(38, 0),
            "src_block_id": pa.string(),
            "dst_block_id": pa.string(),
        },
    )


def _join_block_lookup(edges: pa.Table, lookup: pa.Table, *, table_key: str) -> pa.Table:
    if lookup.num_rows == 0:
        return edges
    edge_keys = ["function_goid_h128", "src_block_id", "dst_block_id"]
    edge_precheck = finalize_join_keys(
        edges,
        required_non_null=edge_keys,
        key_fields=edge_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        edge_precheck,
        table_key=table_key,
        target_name=CPG_TARGET_NAME,
        join_keys=edge_keys,
    )
    edges = edge_precheck.good
    src_lookup = rename_table_columns(
        lookup,
        {
            "block_id": "src_block_id",
            "block_idx": "src_block_idx",
            "rel_path": "src_rel_path",
            "repo": "src_repo",
            "commit": "src_commit",
        },
    )
    edges = _join_ready(edges, table_key=table_key)
    src_lookup = _join_ready(src_lookup, table_key=CFG_BLOCKS_TABLE_KEY)
    dst_lookup = rename_table_columns(
        lookup,
        {
            "block_id": "dst_block_id",
            "block_idx": "dst_block_idx",
            "rel_path": "dst_rel_path",
            "repo": "dst_repo",
            "commit": "dst_commit",
        },
    )
    dst_lookup = _join_ready(dst_lookup, table_key=CFG_BLOCKS_TABLE_KEY)
    edge_project: dict[str, Expression] = {}
    for name in edges.column_names:
        if name == "function_goid_h128":
            edge_project[name] = E.cast(E.field(name), "decimal128(38,0)")
        elif name in {"src_block_id", "dst_block_id"}:
            edge_project[name] = E.cast(E.field(name), "string")
        else:
            edge_project[name] = E.field(name)
    edge_plan = build_table_plan(
        table=edges,
        options=TablePlanOptions(projection=edge_project),
    )
    src_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "src_block_id": E.cast(E.field("src_block_id"), "string"),
        "src_block_idx": E.field("src_block_idx"),
        "src_rel_path": E.field("src_rel_path"),
        "src_repo": E.field("src_repo"),
        "src_commit": E.field("src_commit"),
    }
    src_plan = build_table_plan(
        table=src_lookup,
        options=TablePlanOptions(projection=src_project),
    )
    joined = edge_plan.hash_join(
        right=src_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128", "src_block_id"],
            right_keys=["function_goid_h128", "src_block_id"],
            how="left outer",
            left_output=list(edge_project.keys()),
            right_output=["src_block_idx", "src_rel_path", "src_repo", "src_commit"],
        ),
    )
    dst_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "dst_block_id": E.cast(E.field("dst_block_id"), "string"),
        "dst_block_idx": E.field("dst_block_idx"),
        "dst_rel_path": E.field("dst_rel_path"),
        "dst_repo": E.field("dst_repo"),
        "dst_commit": E.field("dst_commit"),
    }
    dst_plan = build_table_plan(
        table=dst_lookup,
        options=TablePlanOptions(projection=dst_project),
    )
    joined = joined.hash_join(
        right=dst_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128", "dst_block_id"],
            right_keys=["function_goid_h128", "dst_block_id"],
            how="left outer",
            left_output=[
                *edge_project.keys(),
                "src_block_idx",
                "src_rel_path",
                "src_repo",
                "src_commit",
            ],
            right_output=["dst_block_idx", "dst_rel_path", "dst_repo", "dst_commit"],
        ),
    )
    return _plan_to_table(joined, use_threads=True)


def _join_block_anchors(edges: pa.Table, anchors: pa.Table, *, table_key: str) -> pa.Table:
    if anchors.num_rows == 0:
        return edges
    if "src_block_idx" not in edges.column_names or "dst_block_idx" not in edges.column_names:
        return edges
    edge_keys = ["function_goid_h128", "src_block_idx", "dst_block_idx"]
    edge_precheck = finalize_join_keys(
        edges,
        required_non_null=edge_keys,
        key_fields=edge_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        edge_precheck,
        table_key=table_key,
        target_name=CPG_TARGET_NAME,
        join_keys=edge_keys,
    )
    edges = edge_precheck.good
    src_anchor = rename_table_columns(
        anchors,
        {"block_idx": "src_block_idx", "cpg_node_id": "src_cpg_node_id"},
    )
    edges = _join_ready(edges, table_key=table_key)
    src_anchor = _join_ready(src_anchor, table_key=CFG_BLOCKS_TABLE_KEY)
    dst_anchor = rename_table_columns(
        anchors,
        {"block_idx": "dst_block_idx", "cpg_node_id": "dst_cpg_node_id"},
    )
    dst_anchor = _join_ready(dst_anchor, table_key=CFG_BLOCKS_TABLE_KEY)
    edge_project: dict[str, Expression] = {}
    for name in edges.column_names:
        if name == "function_goid_h128":
            edge_project[name] = E.cast(E.field(name), "decimal128(38,0)")
        elif name in {"src_block_idx", "dst_block_idx"}:
            edge_project[name] = E.cast(E.field(name), "int64")
        else:
            edge_project[name] = E.field(name)
    edge_plan = build_table_plan(
        table=edges,
        options=TablePlanOptions(projection=edge_project),
    )
    src_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "src_block_idx": E.cast(E.field("src_block_idx"), "int64"),
        "src_cpg_node_id": E.field("src_cpg_node_id"),
    }
    src_plan = build_table_plan(
        table=src_anchor,
        options=TablePlanOptions(projection=src_project),
    )
    joined = edge_plan.hash_join(
        right=src_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128", "src_block_idx"],
            right_keys=["function_goid_h128", "src_block_idx"],
            how="left outer",
            left_output=list(edge_project.keys()),
            right_output=["src_cpg_node_id"],
        ),
    )
    dst_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "dst_block_idx": E.cast(E.field("dst_block_idx"), "int64"),
        "dst_cpg_node_id": E.field("dst_cpg_node_id"),
    }
    dst_plan = build_table_plan(
        table=dst_anchor,
        options=TablePlanOptions(projection=dst_project),
    )
    joined = joined.hash_join(
        right=dst_plan,
        spec=HashJoinSpec(
            left_keys=["function_goid_h128", "dst_block_idx"],
            right_keys=["function_goid_h128", "dst_block_idx"],
            how="left outer",
            left_output=[*edge_project.keys(), "src_cpg_node_id"],
            right_output=["dst_cpg_node_id"],
        ),
    )
    return _plan_to_table(joined, use_threads=True)


def _coalesce_rel_path(
    table: pa.Table,
    src_col: str,
    dst_col: str,
    *,
    result_type: pa.DataType | None = None,
) -> pa.Array | pa.ChunkedArray:
    if src_col not in table.column_names or dst_col not in table.column_names:
        null_type = result_type or pa.string()
        return pa.nulls(table.num_rows, type=null_type)
    src = table.column(src_col)
    dst = table.column(dst_col)
    src_valid = is_valid_mask(src)
    result = array_from_compute("if_else", [src_valid, src, dst])
    if result is None:
        msg = "Arrow compute if_else did not return an array."
        raise TypeError(msg)
    if result_type is None:
        return result
    return cast_array(result, result_type, safe=False)


def _upsert_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index == -1:
        return table.append_column(name, values)
    return table.set_column(index, name, values)


def _rename_if_present(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    if not mapping:
        return table
    if not any(name in table.column_names for name in mapping):
        return table
    return rename_table_columns(table, mapping)


def _join_ready(table: pa.Table, *, table_key: str | None = None) -> pa.Table:
    allowlist = _join_safe_allowlist(table_key)
    normalized = normalize_table_for_join(table, allowed_columns=allowlist)
    return join_safe_projection(normalized, allowed_columns=allowlist)


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


def _assign_ordinals(
    table: pa.Table,
    *,
    table_key: str,
    columns: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0:
        if "ordinal" in table.column_names:
            return table
        return table.append_column("ordinal", pa.nulls(0, type=pa.int64()))
    seen: set[str] = set()
    hash_columns: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        hash_columns.append(column)
    ordinals = cpg_edge_ordinals(table, table_key=table_key, columns=hash_columns)
    return _upsert_column(table, "ordinal", ordinals)


def _filter_valid_edges(table: pa.Table) -> pa.Table:
    required = {"src_cpg_node_id", "dst_cpg_node_id"}
    if not required.issubset(set(table.column_names)):
        return table

    expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
    return plan_filter_or_fallback(table, expr)


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    if "cpg_node_id" not in table.column_names:
        return table
    return plan_filter_or_fallback(table, is_valid_expr("cpg_node_id"))


__all__ = [
    "CdgEdgeDiagnostics",
    "CfgBlockDiagnostics",
    "CfgEdgeDiagnostics",
    "DfgEdgeDiagnostics",
    "cpg2_edges__cdg_edges",
    "cpg2_edges__cfg_edges",
    "cpg2_edges__dfg_edges",
    "cpg2_nodes__cfg_blocks",
]
