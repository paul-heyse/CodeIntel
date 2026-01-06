"""Flow plane CPG nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import rename_table_columns, select_table_columns
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal
from codeintel.build.tabular.arrow_ops import ArrowJoinSpec, arrow_join_tables
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.serialization.payload import encode_payload

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
    goid_ctx = _goid_context(goids)
    joined = arrow_join_tables(
        normalized_blocks,
        goid_ctx,
        spec=ArrowJoinSpec(on=["function_goid_h128"], how="left"),
    )
    anchors = build_anchor_map(
        normalized_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        pk_columns=identity_keys(CFG_BLOCKS_TABLE_KEY),
        include_source_pk_json=True,
    )
    joined = arrow_join_tables(
        joined,
        anchors,
        spec=ArrowJoinSpec(on=["function_goid_h128", "block_idx"], how="left"),
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
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = _join_block_anchors(joined, anchors)
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path")
    joined = joined.append_column("rel_path", rel_path)
    ordinals = [
        cpg_edge_ordinal(
            "graph.cfg_edges",
            {
                "function_goid_h128": row.get("function_goid_h128"),
                "src_block_id": row.get("src_block_id"),
                "dst_block_id": row.get("dst_block_id"),
                "edge_kind": row.get("edge_kind"),
            },
        )
        for row in joined.select(
            ["function_goid_h128", "src_block_id", "dst_block_id", "edge_kind"]
        ).to_pylist()
    ]
    extras = [
        _payload_bytes({"cfg_edge_kind": row.get("edge_kind")})
        for row in joined.select(["edge_kind"]).to_pylist()
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    joined = append_constant_columns(joined, {"edge_kind": "CFG", "edge_layer": "FLOW"})
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
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = _join_block_anchors(joined, anchors)
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path")
    joined = joined.append_column("rel_path", rel_path)
    ordinals = [
        cpg_edge_ordinal(
            "graph.dfg_edges",
            {
                "function_goid_h128": row.get("function_goid_h128"),
                "src_block_id": row.get("src_block_id"),
                "dst_block_id": row.get("dst_block_id"),
                "src_var": row.get("src_var"),
                "dst_var": row.get("dst_var"),
            },
        )
        for row in joined.select(
            [
                "function_goid_h128",
                "src_block_id",
                "dst_block_id",
                "src_var",
                "dst_var",
            ]
        ).to_pylist()
    ]
    extras = [
        _payload_bytes(
            {
                "src_var": row.get("src_var"),
                "dst_var": row.get("dst_var"),
                "edge_kind": row.get("edge_kind"),
                "via_phi": row.get("via_phi"),
                "use_kind": row.get("use_kind"),
            }
        )
        for row in joined.select(
            ["src_var", "dst_var", "edge_kind", "via_phi", "use_kind"]
        ).to_pylist()
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    joined = append_constant_columns(joined, {"edge_kind": "DFG", "edge_layer": "FLOW"})
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
    lookup = _cfg_block_lookup(cfg_blocks, goids)
    anchors = _cfg_block_anchor(cfg_blocks)
    joined = _join_block_lookup(normalized_edges, lookup)
    joined = _join_block_anchors(joined, anchors)
    rel_path = _coalesce_rel_path(joined, "src_rel_path", "dst_rel_path")
    joined = joined.append_column("rel_path", rel_path)
    ordinals = [
        cpg_edge_ordinal(
            "graph.cdg_edges",
            {
                "function_goid_h128": row.get("function_goid_h128"),
                "src_block_id": row.get("src_block_id"),
                "dst_block_id": row.get("dst_block_id"),
                "via_succ_block_id": row.get("via_succ_block_id"),
            },
        )
        for row in joined.select(
            [
                "function_goid_h128",
                "src_block_id",
                "dst_block_id",
                "via_succ_block_id",
            ]
        ).to_pylist()
    ]
    extras = [
        _payload_bytes(
            {
                "via_succ_block_id": row.get("via_succ_block_id"),
                "via_edge_kind": row.get("via_edge_kind"),
            }
        )
        for row in joined.select(["via_succ_block_id", "via_edge_kind"]).to_pylist()
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    edge_kinds = [row.get("edge_kind") or "CDG" for row in joined.select(["edge_kind"]).to_pylist()]
    joined = joined.append_column("edge_kind", pa.array(edge_kinds, type=pa.string()))
    joined = append_constant_columns(joined, {"edge_layer": "FLOW"})
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
    goid_ctx = _goid_context(goids)
    joined = arrow_join_tables(
        normalized_blocks,
        goid_ctx,
        spec=ArrowJoinSpec(on=["function_goid_h128"], how="left"),
    )
    joined = rename_table_columns(joined, {"file_path": "rel_path"})
    return joined.select(
        ["function_goid_h128", "block_id", "block_idx", "rel_path", "repo", "commit"]
    )


def _cfg_block_anchor(cfg_blocks: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(
        cfg_blocks,
        table_key=CFG_BLOCKS_TABLE_KEY,
        casts={"function_goid_h128": pa.decimal128(38, 0)},
    )
    return build_anchor_map(
        normalized,
        table_key=CFG_BLOCKS_TABLE_KEY,
        pk_columns=identity_keys(CFG_BLOCKS_TABLE_KEY),
        include_source_pk_json=False,
    )


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
    joined = arrow_join_tables(
        edges,
        src_lookup,
        spec=ArrowJoinSpec(on=["function_goid_h128", "src_block_id"], how="left"),
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
    return arrow_join_tables(
        joined,
        dst_lookup,
        spec=ArrowJoinSpec(on=["function_goid_h128", "dst_block_id"], how="left"),
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
    joined = arrow_join_tables(
        edges,
        src_anchor,
        spec=ArrowJoinSpec(on=["function_goid_h128", "src_block_idx"], how="left"),
    )
    dst_anchor = rename_table_columns(
        anchors,
        {"block_idx": "dst_block_idx", "cpg_node_id": "dst_cpg_node_id"},
    )
    return arrow_join_tables(
        joined,
        dst_anchor,
        spec=ArrowJoinSpec(on=["function_goid_h128", "dst_block_idx"], how="left"),
    )


def _coalesce_rel_path(table: pa.Table, src_col: str, dst_col: str) -> pa.Array:
    if src_col not in table.column_names or dst_col not in table.column_names:
        return pa.nulls(table.num_rows)
    src = table.column(src_col)
    dst = table.column(dst_col)
    src_valid = is_valid_mask(src)
    return pc.call_function("if_else", [src_valid, src, dst])


def _filter_valid_edges(table: pa.Table) -> pa.Table:
    mask = and_kleene(
        is_valid_mask(table.column("src_cpg_node_id")),
        is_valid_mask(table.column("dst_cpg_node_id")),
    )
    return safe_filter(table, mask)


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    mask = is_valid_mask(table.column("cpg_node_id"))
    return safe_filter(table, mask)


def _payload_bytes(values: dict[str, object]) -> bytes:
    encoded = encode_payload(values)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


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
