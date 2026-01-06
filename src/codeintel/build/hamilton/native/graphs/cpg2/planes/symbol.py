"""SCIP symbol plane CPG edges."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.graphs.assembly import rename_table_columns
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

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"


@dataclass(frozen=True)
class SymbolRelationshipDiagnostics:
    """Diagnostics for SCIP symbol relationship edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class SymbolGoidDiagnostics:
    """Diagnostics for SCIP symbol to GOID edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


def cpg2_edges__scip_symbol_relationships(
    symbol_rels: pa.Table,
    scip_symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from SCIP symbol relationships.

    Returns
    -------
    pyarrow.Table
        CPG edges for SCIP symbol relationships.
    """
    if symbol_rels.num_rows == 0:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    required = {"repo", "commit", "symbol", "related_symbol", "relationship_kind"}
    if not required.issubset(set(symbol_rels.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    rels = _normalize_symbol_rels(symbol_rels)
    anchors = _symbol_anchor_map(scip_symbols)
    src_anchor = rename_table_columns(anchors, {"cpg_node_id": "src_cpg_node_id"})
    joined = arrow_join_tables(
        rels,
        src_anchor,
        spec=ArrowJoinSpec(on=["repo", "commit", "symbol"], how="left"),
    )
    dst_anchor = rename_table_columns(
        anchors,
        {
            "symbol": "related_symbol",
            "cpg_node_id": "dst_cpg_node_id",
        },
    )
    joined = arrow_join_tables(
        joined,
        dst_anchor,
        spec=ArrowJoinSpec(on=["repo", "commit", "related_symbol"], how="left"),
    )
    ordinals = [
        cpg_edge_ordinal(
            "core.scip_symbol_relationships",
            {
                "symbol": row.get("symbol"),
                "related_symbol": row.get("related_symbol"),
                "relationship_kind": row.get("relationship_kind"),
            },
        )
        for row in joined.select(["symbol", "related_symbol", "relationship_kind"]).to_pylist()
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = append_constant_columns(
        joined,
        {
            "edge_layer": "SYMBOL",
            "rel_path": None,
            "extras_json": None,
        },
    )
    joined = rename_table_columns(joined, {"relationship_kind": "edge_kind"})
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
        diagnostics["scip_symbol_relationships"] = SymbolRelationshipDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__scip_symbol_goid_xref(
    symbol_goid: pa.Table,
    scip_symbols: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from SCIP symbol to GOID crosswalks.

    Returns
    -------
    pyarrow.Table
        CPG edges for symbol-to-GOID mappings.
    """
    if symbol_goid.num_rows == 0:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    required = {"repo", "commit", "scip_symbol", "goid_h128"}
    if not required.issubset(set(symbol_goid.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    goid_rows = _normalize_symbol_goid(symbol_goid)
    mask = is_valid_mask(goid_rows.column("goid_h128"))
    goid_rows = safe_filter(goid_rows, mask)
    symbol_anchors = rename_table_columns(
        _symbol_anchor_map(scip_symbols),
        {"symbol": "scip_symbol", "cpg_node_id": "src_cpg_node_id"},
    )
    goid_anchors = rename_table_columns(
        _goid_anchor_map(goids),
        {"cpg_node_id": "dst_cpg_node_id"},
    )
    joined = arrow_join_tables(
        goid_rows,
        symbol_anchors,
        spec=ArrowJoinSpec(on=["repo", "commit", "scip_symbol"], how="left"),
    )
    joined = arrow_join_tables(
        joined,
        goid_anchors,
        spec=ArrowJoinSpec(on=["goid_h128"], how="left"),
    )
    ordinals = [
        cpg_edge_ordinal(
            "core.scip_symbol_goid_xref",
            {"scip_symbol": row.get("scip_symbol"), "goid_h128": row.get("goid_h128")},
        )
        for row in joined.select(["scip_symbol", "goid_h128"]).to_pylist()
    ]
    extras = [
        _encode_extras_payload(
            {
                "def_rel_path": row.get("def_rel_path"),
                "def_start_line": row.get("def_start_line"),
                "def_start_col": row.get("def_start_col"),
                "def_end_line": row.get("def_end_line"),
                "def_end_col": row.get("def_end_col"),
            }
        )
        for row in joined.select(
            [
                "def_rel_path",
                "def_start_line",
                "def_start_col",
                "def_end_line",
                "def_end_col",
            ]
        ).to_pylist()
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    joined = append_constant_columns(joined, {"edge_kind": "RESOLVES_TO", "edge_layer": "SYMBOL"})
    joined = rename_table_columns(joined, {"def_rel_path": "rel_path"})
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
        diagnostics["scip_symbol_goid"] = SymbolGoidDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def _symbol_anchor_map(scip_symbols: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(scip_symbols, table_key=SCIP_SYMBOLS_TABLE_KEY)
    return build_anchor_map(
        normalized,
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        pk_columns=identity_keys(SCIP_SYMBOLS_TABLE_KEY),
        include_source_pk_json=False,
    )


def _goid_anchor_map(goids: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(goids, table_key=GOIDS_TABLE_KEY)
    return build_anchor_map(
        normalized,
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=False,
    )


def _normalize_symbol_rels(table: pa.Table) -> pa.Table:
    return canonicalize_for_table(
        table,
        table_key="core.scip_symbol_relationships",
        casts={
            "repo": pa.string(),
            "commit": pa.string(),
            "symbol": pa.string(),
            "related_symbol": pa.string(),
            "relationship_kind": pa.string(),
        },
    )


def _normalize_symbol_goid(table: pa.Table) -> pa.Table:
    return canonicalize_for_table(
        table,
        table_key="core.scip_symbol_goid_xref",
        casts={
            "repo": pa.string(),
            "commit": pa.string(),
            "scip_symbol": pa.string(),
            "goid_h128": pa.decimal128(38, 0),
        },
    )


def _filter_valid_edges(table: pa.Table) -> pa.Table:
    mask = and_kleene(
        is_valid_mask(table.column("src_cpg_node_id")),
        is_valid_mask(table.column("dst_cpg_node_id")),
    )
    return safe_filter(table, mask)


def _encode_extras_payload(values: dict[str, object]) -> bytes | None:
    encoded = encode_payload(values)
    if encoded is None:
        return None
    return encoded


__all__ = [
    "SymbolGoidDiagnostics",
    "SymbolRelationshipDiagnostics",
    "cpg2_edges__scip_symbol_goid_xref",
    "cpg2_edges__scip_symbol_relationships",
]
