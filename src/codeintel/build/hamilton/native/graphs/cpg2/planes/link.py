"""Link plane CPG nodes and edges."""

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
from codeintel.build.tabular.compute_helpers import safe_filter, safe_filter_expr
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.serialization.payload import encode_payload

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
GOIDS_TABLE_KEY = "core.goids"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"

_EXPR_TYPE = getattr(pc, "Expression", None)


@dataclass(frozen=True)
class CallGraphDiagnostics:
    """Diagnostics for call graph edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class ImportGraphDiagnostics:
    """Diagnostics for import graph edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class ImportModuleDiagnostics:
    """Diagnostics for import module CPG nodes."""

    total_rows: int
    resolved_rows: int
    dropped_rows: int


def cpg2_nodes__import_modules(
    import_modules: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes for import module inventory.

    Returns
    -------
    pyarrow.Table
        CPG nodes for import modules.
    """
    required = {"repo", "commit", "module"}
    if not required.issubset(set(import_modules.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized = canonicalize_for_table(import_modules, table_key=IMPORT_MODULES_TABLE_KEY)
    normalized = normalize_table_for_join(normalized)
    anchors = build_anchor_map(
        normalized,
        table_key=IMPORT_MODULES_TABLE_KEY,
        pk_columns=identity_keys(IMPORT_MODULES_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = normalize_table_for_join(anchors)
    normalized = normalize_table_for_join(normalized)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "module"], how="left")
    filter_expr = join_filter_expr(
        left=normalized,
        right=anchors,
        spec=join_spec,
        clause=JoinFilterClause(
            field="cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        normalized,
        anchors,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        normalized,
        anchors,
        spec=join_spec,
        options=join_options,
    )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "MODULE",
            "source_table_key": IMPORT_MODULES_TABLE_KEY,
            "rel_path": None,
            "start_byte": None,
            "end_byte": None,
            "extras_json": None,
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
            "extras_json",
        ]
    )
    filtered = _filter_valid_nodes(selected)
    if diagnostics is not None:
        diagnostics["import_modules"] = ImportModuleDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__call_graph_edges(
    call_edges: pa.Table,
    goids: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from the call graph.

    Returns
    -------
    pyarrow.Table
        CPG edges for call graph links.
    """
    required = {"repo", "commit", "caller_goid_h128", "callee_goid_h128", "callsite_path"}
    if not required.issubset(set(call_edges.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    normalized_edges = canonicalize_for_table(call_edges, table_key="graph.call_graph_edges")
    normalized_edges = normalize_table_for_join(normalized_edges)
    anchor_base = build_anchor_map(
        canonicalize_for_table(goids, table_key=GOIDS_TABLE_KEY),
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=False,
    )
    anchor_base = normalize_table_for_join(anchor_base)
    src_anchor = rename_table_columns(
        anchor_base,
        {"goid_h128": "caller_goid_h128", "cpg_node_id": "src_cpg_node_id"},
    )
    src_anchor = normalize_table_for_join(src_anchor)
    join_spec = ArrowJoinSpec(on=["caller_goid_h128"], how="left")
    filter_expr = join_filter_expr(
        left=normalized_edges,
        right=src_anchor,
        spec=join_spec,
        clause=JoinFilterClause(
            field="src_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        normalized_edges,
        src_anchor,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        normalized_edges,
        src_anchor,
        spec=join_spec,
        options=join_options,
    )
    dst_anchor = rename_table_columns(
        anchor_base,
        {"goid_h128": "callee_goid_h128", "cpg_node_id": "dst_cpg_node_id"},
    )
    joined = normalize_table_for_join(joined)
    dst_anchor = normalize_table_for_join(dst_anchor)
    join_spec = ArrowJoinSpec(on=["callee_goid_h128"], how="left")
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
            "graph.call_graph_edges",
            {
                "caller_goid_h128": row.get("caller_goid_h128"),
                "callee_goid_h128": row.get("callee_goid_h128"),
                "callsite_path": row.get("callsite_path"),
                "callsite_line": row.get("callsite_line"),
                "callsite_col": row.get("callsite_col"),
            },
        )
        for row in iter_rows(
            joined,
            [
                "caller_goid_h128",
                "callee_goid_h128",
                "callsite_path",
                "callsite_line",
                "callsite_col",
            ],
        )
    ]
    extras = [
        _payload_bytes(
            {
                "resolved_via": row.get("resolved_via"),
                "confidence": row.get("confidence"),
                "kind": row.get("kind"),
            }
        )
        for row in iter_rows(joined, ["resolved_via", "confidence", "kind"])
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    joined = append_constant_columns(joined, {"edge_kind": "CALLS", "edge_layer": "FLOW"})
    joined = rename_table_columns(joined, {"callsite_path": "rel_path"})
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
        diagnostics["call_graph"] = CallGraphDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__import_graph_edges(
    import_edges: pa.Table,
    import_modules: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from module-level import graph edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for import graph links.
    """
    required = {"repo", "commit", "src_module", "dst_module"}
    if not required.issubset(set(import_edges.column_names)):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    normalized_edges = canonicalize_for_table(import_edges, table_key="graph.import_graph_edges")
    normalized_edges = normalize_table_for_join(normalized_edges)
    anchor_base = build_anchor_map(
        canonicalize_for_table(
            select_table_columns(import_modules, ["repo", "commit", "module"]),
            table_key=IMPORT_MODULES_TABLE_KEY,
        ),
        table_key=IMPORT_MODULES_TABLE_KEY,
        pk_columns=identity_keys(IMPORT_MODULES_TABLE_KEY),
        include_source_pk_json=False,
    )
    anchor_base = normalize_table_for_join(anchor_base)
    src_anchor = rename_table_columns(
        anchor_base,
        {"module": "src_module", "cpg_node_id": "src_cpg_node_id"},
    )
    src_anchor = normalize_table_for_join(src_anchor)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "src_module"], how="left")
    filter_expr = join_filter_expr(
        left=normalized_edges,
        right=src_anchor,
        spec=join_spec,
        clause=JoinFilterClause(
            field="src_cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        normalized_edges,
        src_anchor,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        normalized_edges,
        src_anchor,
        spec=join_spec,
        options=join_options,
    )
    dst_anchor = rename_table_columns(
        anchor_base,
        {"module": "dst_module", "cpg_node_id": "dst_cpg_node_id"},
    )
    joined = normalize_table_for_join(joined)
    dst_anchor = normalize_table_for_join(dst_anchor)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "dst_module"], how="left")
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
            "graph.import_graph_edges",
            {
                "src_module": row.get("src_module"),
                "dst_module": row.get("dst_module"),
                "cycle_group": row.get("cycle_group"),
            },
        )
        for row in iter_rows(joined, ["src_module", "dst_module", "cycle_group"])
    ]
    extras = [
        _payload_bytes(
            {
                "src_fan_out": row.get("src_fan_out"),
                "dst_fan_in": row.get("dst_fan_in"),
                "cycle_group": row.get("cycle_group"),
                "module_layer": row.get("module_layer"),
            }
        )
        for row in iter_rows(
            joined,
            ["src_fan_out", "dst_fan_in", "cycle_group", "module_layer"],
        )
    ]
    joined = joined.append_column("ordinal", pa.array(ordinals, type=pa.int64()))
    joined = joined.append_column("extras_json", pa.array(extras, type=pa.binary()))
    joined = append_constant_columns(joined, {"edge_kind": "IMPORTS", "edge_layer": "SYMBOL"})
    joined = append_constant_columns(joined, {"rel_path": None})
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
        diagnostics["import_graph"] = ImportGraphDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=selected.num_rows - filtered.num_rows,
        )
    return filtered


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


def _payload_bytes(values: dict[str, object]) -> bytes:
    encoded = encode_payload(values)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


__all__ = [
    "CallGraphDiagnostics",
    "ImportGraphDiagnostics",
    "ImportModuleDiagnostics",
    "cpg2_edges__call_graph_edges",
    "cpg2_edges__import_graph_edges",
    "cpg2_nodes__import_modules",
]
