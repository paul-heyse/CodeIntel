"""Flow plane CPG nodes and edges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
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
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    JoinFilterClause,
    arrow_join_tables,
    build_join_options,
    join_filter_expr,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import cast_array, safe_filter, safe_filter_expr
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.build.tabular.conversion import table_to_frame
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
    join_spec = ArrowJoinSpec(on=["function_goid_h128"], how="left")
    join_options = build_join_options(normalized_blocks, goid_ctx, normalize_inputs=False)
    joined = arrow_join_tables(
        normalized_blocks,
        goid_ctx,
        spec=join_spec,
        options=join_options,
    )
    anchors = build_anchor_map(
        normalized_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        pk_columns=identity_keys(CFG_BLOCKS_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = normalize_table_for_join(anchors)
    joined = normalize_table_for_join(joined)
    join_spec = ArrowJoinSpec(on=["function_goid_h128", "block_idx"], how="left")
    join_options = build_join_options(joined, anchors, normalize_inputs=False)
    joined = arrow_join_tables(
        joined,
        anchors,
        spec=join_spec,
        options=join_options,
    )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "CFG_BLOCK",
            "source_table_key": CFG_BLOCKS_TABLE_KEY,
            "start_byte": None,
            "end_byte": None,
            "extras_json": None,
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
            "extras_json",
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
    joined = joined.append_column("repo", repo)
    joined = joined.append_column("commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "CFG", "edge_layer": "FLOW", "extras_json": None},
    )
    joined = _assign_ordinals(
        joined,
        group_cols=[
            "repo",
            "commit",
            "src_cpg_node_id",
            "dst_cpg_node_id",
            "edge_kind",
            "edge_layer",
        ],
        sort_cols=[
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "cfg_edge_kind",
        ],
    )
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
            "extras_json",
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
    joined = joined.append_column("repo", repo)
    joined = joined.append_column("commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "DFG", "edge_layer": "FLOW", "extras_json": None},
    )
    joined = _assign_ordinals(
        joined,
        group_cols=[
            "repo",
            "commit",
            "src_cpg_node_id",
            "dst_cpg_node_id",
            "edge_kind",
            "edge_layer",
        ],
        sort_cols=[
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "src_var",
            "dst_var",
            "dfg_edge_kind",
            "via_phi",
            "use_kind",
        ],
    )
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
            "extras_json",
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
    joined = joined.append_column("repo", repo)
    joined = joined.append_column("commit", commit)
    joined = append_constant_columns(
        joined,
        {"edge_kind": "CDG", "edge_layer": "FLOW", "extras_json": None},
    )
    joined = _assign_ordinals(
        joined,
        group_cols=[
            "repo",
            "commit",
            "src_cpg_node_id",
            "dst_cpg_node_id",
            "edge_kind",
            "edge_layer",
        ],
        sort_cols=[
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "via_succ_block_id",
            "cdg_edge_kind",
            "via_edge_kind",
        ],
    )
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
            "extras_json",
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
    join_spec = ArrowJoinSpec(on=["function_goid_h128"], how="left")
    join_options = build_join_options(normalized_blocks, goid_ctx, normalize_inputs=False)
    joined = arrow_join_tables(
        normalized_blocks,
        goid_ctx,
        spec=join_spec,
        options=join_options,
    )
    joined = rename_table_columns(joined, {"file_path": "rel_path"})
    return normalize_table_for_join(
        joined.select(["function_goid_h128", "block_id", "block_idx", "rel_path", "repo", "commit"])
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
    join_spec = ArrowJoinSpec(on=["function_goid_h128", "src_block_id"], how="left")
    join_options = build_join_options(edges, src_lookup, normalize_inputs=False)
    joined = arrow_join_tables(
        edges,
        src_lookup,
        spec=join_spec,
        options=join_options,
    )
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
    joined = normalize_table_for_join(joined)
    dst_lookup = normalize_table_for_join(dst_lookup)
    join_spec = ArrowJoinSpec(on=["function_goid_h128", "dst_block_id"], how="left")
    join_options = build_join_options(joined, dst_lookup, normalize_inputs=False)
    return arrow_join_tables(
        joined,
        dst_lookup,
        spec=join_spec,
        options=join_options,
    )


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
    join_spec = ArrowJoinSpec(on=["function_goid_h128", "src_block_idx"], how="left")
    filter_expr = join_filter_expr(
        left=edges,
        right=src_anchor,
        spec=join_spec,
        clause=JoinFilterClause(
            field="src_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        edges,
        src_anchor,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        edges,
        src_anchor,
        spec=join_spec,
        options=join_options,
    )
    dst_anchor = rename_table_columns(
        anchors,
        {"block_idx": "dst_block_idx", "cpg_node_id": "dst_cpg_node_id"},
    )
    joined = normalize_table_for_join(joined)
    dst_anchor = normalize_table_for_join(dst_anchor)
    join_spec = ArrowJoinSpec(on=["function_goid_h128", "dst_block_idx"], how="left")
    filter_expr = join_filter_expr(
        left=joined,
        right=dst_anchor,
        spec=join_spec,
        clause=JoinFilterClause(
            field="dst_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        joined,
        dst_anchor,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    return arrow_join_tables(
        joined,
        dst_anchor,
        spec=join_spec,
        options=join_options,
    )


def _coalesce_rel_path(
    table: pa.Table,
    src_col: str,
    dst_col: str,
    *,
    result_type: pa.DataType | None = None,
) -> pa.Array:
    if src_col not in table.column_names or dst_col not in table.column_names:
        null_type = result_type or pa.string()
        return pa.nulls(table.num_rows, type=null_type)
    src = table.column(src_col)
    dst = table.column(dst_col)
    src_valid = is_valid_mask(src)
    result = pc.call_function("if_else", [src_valid, src, dst])
    if result_type is None:
        return result
    return cast_array(result, result_type, safe=False)


def _rename_if_present(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    if not mapping:
        return table
    if not any(name in table.column_names for name in mapping):
        return table
    return rename_table_columns(table, mapping)


def _assign_ordinals(
    table: pa.Table,
    *,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0:
        if "ordinal" in table.column_names:
            return table
        return table.append_column("ordinal", pa.nulls(0, type=pa.int64()))
    frame = table_to_frame(table)
    sort_keys = [name for name in sort_cols if name in frame.columns]
    if sort_keys:
        frame = frame.sort(sort_keys)
    group_keys = [name for name in group_cols if name in frame.columns]
    if group_keys:
        frame = frame.with_columns(
            pl.int_range(0, pl.len()).over(group_keys).cast(pl.Int64).alias("ordinal")
        )
    else:
        frame = frame.with_columns(pl.int_range(0, pl.len()).cast(pl.Int64).alias("ordinal"))
    return frame.to_arrow()


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
