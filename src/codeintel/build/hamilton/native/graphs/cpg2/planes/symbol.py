"""SCIP symbol plane CPG edges."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import rename_table_columns
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    JoinFilterClause,
    arrow_join_tables,
    build_join_options,
    iter_rows,
    join_filter_expr,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.serialization.payload import encode_payload

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"

_EXPR_TYPE = getattr(pc, "Expression", None)


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
    rels = normalize_table_for_join(_normalize_symbol_rels(symbol_rels))
    anchors = normalize_table_for_join(_symbol_anchor_map(scip_symbols))
    src_anchor = rename_table_columns(anchors, {"cpg_node_id": "src_cpg_node_id"})
    src_anchor = normalize_table_for_join(src_anchor)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "symbol"], how="left")
    filter_expr = join_filter_expr(
        left=rels,
        right=src_anchor,
        spec=join_spec,
        clause=JoinFilterClause(
            field="src_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        rels,
        src_anchor,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        rels,
        src_anchor,
        spec=join_spec,
        options=join_options,
    )
    dst_anchor = rename_table_columns(
        anchors,
        {
            "symbol": "related_symbol",
            "cpg_node_id": "dst_cpg_node_id",
        },
    )
    joined = normalize_table_for_join(joined)
    dst_anchor = normalize_table_for_join(dst_anchor)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "related_symbol"], how="left")
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
    joined = arrow_join_tables(
        joined,
        dst_anchor,
        spec=join_spec,
        options=join_options,
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
        for row in iter_rows(joined, ["symbol", "related_symbol", "relationship_kind"])
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
    if "goid_h128" in goid_rows.column_names and _EXPR_TYPE is not None:
        try:
            goid_rows = goid_rows.filter(is_valid_expr("goid_h128"))
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            TypeError,
            ValueError,
        ):
            mask = is_valid_mask(goid_rows.column("goid_h128"))
            goid_rows = safe_filter(goid_rows, mask)
    elif "goid_h128" in goid_rows.column_names:
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
    goid_rows = normalize_table_for_join(goid_rows)
    symbol_anchors = normalize_table_for_join(symbol_anchors)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "scip_symbol"], how="left")
    filter_expr = join_filter_expr(
        left=goid_rows,
        right=symbol_anchors,
        spec=join_spec,
        clause=JoinFilterClause(
            field="src_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        goid_rows,
        symbol_anchors,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        goid_rows,
        symbol_anchors,
        spec=join_spec,
        options=join_options,
    )
    joined = normalize_table_for_join(joined)
    goid_anchors = normalize_table_for_join(goid_anchors)
    join_spec = ArrowJoinSpec(on=["goid_h128"], how="left")
    filter_expr = join_filter_expr(
        left=joined,
        right=goid_anchors,
        spec=join_spec,
        clause=JoinFilterClause(
            field="dst_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        joined,
        goid_anchors,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        joined,
        goid_anchors,
        spec=join_spec,
        options=join_options,
    )
    ordinals = [
        cpg_edge_ordinal(
            "core.scip_symbol_goid_xref",
            {"scip_symbol": row.get("scip_symbol"), "goid_h128": row.get("goid_h128")},
        )
        for row in iter_rows(joined, ["scip_symbol", "goid_h128"])
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
        for row in iter_rows(
            joined,
            [
                "def_rel_path",
                "def_start_line",
                "def_start_col",
                "def_end_line",
                "def_end_col",
            ],
        )
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
    required = {"src_cpg_node_id", "dst_cpg_node_id"}
    if not required.issubset(set(table.column_names)):
        return table
    if _EXPR_TYPE is not None:
        try:
            expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
            return table.filter(expr)
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            TypeError,
            ValueError,
        ):
            pass
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
