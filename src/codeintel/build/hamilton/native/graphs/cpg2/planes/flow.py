"""Flow plane CPG nodes and edges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import (
    rename_table_columns,
    select_table_columns,
)
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinals
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import (
    array_from_compute,
    cast_array,
    safe_filter,
    safe_filter_expr,
)
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.build.tabular.conversion import reader_to_table
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.rows import empty_table_for_table

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
GOIDS_TABLE_KEY = "core.goids"

_EXPR_TYPE = getattr(pc, "Expression", None)


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
    normalized_blocks = normalize_table_for_join(normalized_blocks)
    goid_ctx = normalize_table_for_join(_goid_context(goids))
    block_plan = (
        Plan.table(normalized_blocks)
        .project(
            {
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_idx": E.field("block_idx"),
                "file_path": E.field("file_path"),
            }
        )
        .filter(E.is_valid("function_goid_h128"))
    )
    goid_plan = (
        Plan.table(goid_ctx)
        .project(
            {
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
            }
        )
        .filter(E.is_valid("function_goid_h128"))
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
    anchors = normalize_table_for_join(anchors)
    anchor_plan = (
        Plan.table(anchors)
        .project(
            {
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_idx": E.field("block_idx"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            }
        )
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("block_idx")))
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
    joined_table = reader_to_table(joined.to_reader(use_threads=True))
    if joined_table.num_rows > 0:
        joined_table = joined_table.take(
            stable_sort_indices(
                joined_table,
                sort_keys=[
                    ("repo", "ascending"),
                    ("commit", "ascending"),
                    ("function_goid_h128", "ascending"),
                    ("block_idx", "ascending"),
                ],
            )
        )
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
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = normalize_table_for_join(joined)
    joined = _join_block_anchors(joined, anchors)
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
        joined = joined.take(stable_sort_indices(joined, sort_keys=sort_keys))
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
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = normalize_table_for_join(joined)
    joined = _join_block_anchors(joined, anchors)
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
        joined = joined.take(stable_sort_indices(joined, sort_keys=sort_keys))
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
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = normalize_table_for_join(joined)
    joined = _join_block_anchors(joined, anchors)
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
        joined = joined.take(stable_sort_indices(joined, sort_keys=sort_keys))
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
    normalized_blocks = normalize_table_for_join(normalized_blocks)
    goid_ctx = normalize_table_for_join(_goid_context(goids))
    block_plan = (
        Plan.table(normalized_blocks)
        .project(
            {
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "block_id": E.cast(E.field("block_id"), "string"),
                "block_idx": E.field("block_idx"),
                "file_path": E.field("file_path"),
            }
        )
        .filter(E.is_valid("function_goid_h128"))
    )
    goid_plan = (
        Plan.table(goid_ctx)
        .project(
            {
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
            }
        )
        .filter(E.is_valid("function_goid_h128"))
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
    joined_table = reader_to_table(joined.to_reader(use_threads=True))
    joined_table = rename_table_columns(joined_table, {"file_path": "rel_path"})
    if joined_table.num_rows > 0:
        joined_table = joined_table.take(
            stable_sort_indices(
                joined_table,
                sort_keys=[
                    ("function_goid_h128", "ascending"),
                    ("block_id", "ascending"),
                ],
            )
        )
    return normalize_table_for_join(
        joined_table.select(
            ["function_goid_h128", "block_id", "block_idx", "rel_path", "repo", "commit"]
        )
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
    return normalize_table_for_join(anchors)


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


def _join_block_lookup(edges: pa.Table, lookup: pa.Table) -> pa.Table:
    if lookup.num_rows == 0:
        return edges
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
    edges = normalize_table_for_join(edges)
    src_lookup = normalize_table_for_join(src_lookup)
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
    dst_lookup = normalize_table_for_join(dst_lookup)
    edge_project: dict[str, pc.Expression] = {}
    for name in edges.column_names:
        if name == "function_goid_h128":
            edge_project[name] = E.cast(E.field(name), "decimal128(38,0)")
        elif name in {"src_block_id", "dst_block_id"}:
            edge_project[name] = E.cast(E.field(name), "string")
        else:
            edge_project[name] = E.field(name)
    edge_plan = (
        Plan.table(edges)
        .project(edge_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("src_block_id")))
    )
    src_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "src_block_id": E.cast(E.field("src_block_id"), "string"),
        "src_block_idx": E.field("src_block_idx"),
        "src_rel_path": E.field("src_rel_path"),
        "src_repo": E.field("src_repo"),
        "src_commit": E.field("src_commit"),
    }
    src_plan = (
        Plan.table(src_lookup)
        .project(src_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("src_block_id")))
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
    joined = joined.filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("dst_block_id")))
    dst_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "dst_block_id": E.cast(E.field("dst_block_id"), "string"),
        "dst_block_idx": E.field("dst_block_idx"),
        "dst_rel_path": E.field("dst_rel_path"),
        "dst_repo": E.field("dst_repo"),
        "dst_commit": E.field("dst_commit"),
    }
    dst_plan = (
        Plan.table(dst_lookup)
        .project(dst_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("dst_block_id")))
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
    return reader_to_table(joined.to_reader(use_threads=True))


def _join_block_anchors(edges: pa.Table, anchors: pa.Table) -> pa.Table:
    if anchors.num_rows == 0:
        return edges
    if "src_block_idx" not in edges.column_names or "dst_block_idx" not in edges.column_names:
        return edges
    src_anchor = rename_table_columns(
        anchors,
        {"block_idx": "src_block_idx", "cpg_node_id": "src_cpg_node_id"},
    )
    edges = normalize_table_for_join(edges)
    src_anchor = normalize_table_for_join(src_anchor)
    dst_anchor = rename_table_columns(
        anchors,
        {"block_idx": "dst_block_idx", "cpg_node_id": "dst_cpg_node_id"},
    )
    dst_anchor = normalize_table_for_join(dst_anchor)
    edge_project: dict[str, pc.Expression] = {}
    for name in edges.column_names:
        if name == "function_goid_h128":
            edge_project[name] = E.cast(E.field(name), "decimal128(38,0)")
        elif name in {"src_block_idx", "dst_block_idx"}:
            edge_project[name] = E.cast(E.field(name), "int64")
        else:
            edge_project[name] = E.field(name)
    edge_plan = (
        Plan.table(edges)
        .project(edge_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("src_block_idx")))
    )
    src_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "src_block_idx": E.cast(E.field("src_block_idx"), "int64"),
        "src_cpg_node_id": E.field("src_cpg_node_id"),
    }
    src_plan = (
        Plan.table(src_anchor)
        .project(src_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("src_block_idx")))
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
    joined = joined.filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("dst_block_idx")))
    dst_project = {
        "function_goid_h128": E.cast(E.field("function_goid_h128"), "decimal128(38,0)"),
        "dst_block_idx": E.cast(E.field("dst_block_idx"), "int64"),
        "dst_cpg_node_id": E.field("dst_cpg_node_id"),
    }
    dst_plan = (
        Plan.table(dst_anchor)
        .project(dst_project)
        .filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("dst_block_idx")))
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
    return reader_to_table(joined.to_reader(use_threads=True))


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

    def _edge_mask(target: pa.Table) -> pa.Array | pa.ChunkedArray:
        return and_kleene(
            is_valid_mask(target.column("src_cpg_node_id")),
            is_valid_mask(target.column("dst_cpg_node_id")),
        )

    if _EXPR_TYPE is not None:
        expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
        return safe_filter_expr(table, expr, fallback_mask=_edge_mask)
    return safe_filter(table, _edge_mask(table))


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    if "cpg_node_id" not in table.column_names:
        return table
    if _EXPR_TYPE is not None:
        return safe_filter_expr(
            table,
            is_valid_expr("cpg_node_id"),
            fallback_mask=lambda target: is_valid_mask(target.column("cpg_node_id")),
        )
    return safe_filter(table, is_valid_mask(table.column("cpg_node_id")))


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
