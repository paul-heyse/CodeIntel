"""SCIP symbol plane CPG edges."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyarrow as pa

from codeintel.build.graphs.assembly import rename_table_columns
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
)
from codeintel.build.tabular.compute_masks import is_valid_expr
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.build.tabular.finalize_ops import finalize_join_keys, record_join_precheck_errors
from codeintel.build.tabular.kernels import stable_sort_indices
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan, join_safe_projection
from codeintel.core.columnar.conversion import reader_to_table, table_to_reader
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.primitives import resolve_join_safe_columns

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
SCIP_EXTERNAL_SYMBOLS_TABLE_KEY = "core.scip_external_symbols"
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
    scip_external_symbols: pa.Table,
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
    joined = _symbol_relationship_joins(symbol_rels, scip_symbols, scip_external_symbols)
    ordinals = cpg_edge_ordinals(
        joined,
        table_key="core.scip_symbol_relationships",
        columns=["symbol", "related_symbol", "relationship_kind"],
    )
    extras_kv: list[dict[str, str] | None] = []
    extras_columns = [
        "symbol",
        "related_symbol",
        "relationship_kind",
        "src_cpg_node_id",
        "src_cpg_node_id_ext",
        "dst_cpg_node_id",
        "dst_cpg_node_id_ext",
    ]
    for values in iter_tuples(table_to_reader(joined), columns=extras_columns):
        payload: dict[str, object] = {}
        src_cpg_node_id = values[3]
        src_cpg_node_id_ext = values[4]
        dst_cpg_node_id = values[5]
        dst_cpg_node_id_ext = values[6]
        if src_cpg_node_id is None and src_cpg_node_id_ext is not None:
            payload["src_symbol_origin"] = "external"
        if dst_cpg_node_id is None and dst_cpg_node_id_ext is not None:
            payload["dst_symbol_origin"] = "external"
        extras_kv.append(extras_kv_from_mapping(payload) if payload else None)
    joined = _coalesce_column(joined, "src_cpg_node_id", "src_cpg_node_id_ext")
    joined = _coalesce_column(joined, "dst_cpg_node_id", "dst_cpg_node_id_ext")
    joined = joined.append_column("ordinal", ordinals)
    joined = _upsert_column(
        joined,
        "extras_kv",
        pa.array(extras_kv, type=pa.map_(pa.string(), pa.string())),
    )
    if joined.num_rows > 0:
        ordered_plan = build_table_plan(
            table=joined,
            options=TablePlanOptions(
                order_by=(
                    ("repo", "ascending"),
                    ("commit", "ascending"),
                    ("symbol", "ascending"),
                    ("related_symbol", "ascending"),
                    ("relationship_kind", "ascending"),
                ),
            ),
        )
        joined = _plan_to_table(ordered_plan, use_threads=True)
    joined = append_constant_columns(
        joined,
        {
            "edge_layer": "SYMBOL",
            "rel_path": None,
            "extras": None,
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
            "extras",
            "extras_kv",
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
    joined_table = _symbol_goid_joined_table(symbol_goid, scip_symbols, goids)
    ordinals = cpg_edge_ordinals(
        joined_table,
        table_key="core.scip_symbol_goid_xref",
        columns=["scip_symbol", "goid_h128"],
    )
    extras_fields = [
        field
        for field in [
            "def_rel_path",
            "def_start_line",
            "def_start_col",
            "def_end_line",
            "def_end_col",
        ]
        if field in joined_table.column_names
    ]
    extras_kv: list[dict[str, str] | None] = []
    for values in iter_tuples(table_to_reader(joined_table), columns=extras_fields):
        mapping = dict(zip(extras_fields, values, strict=False))
        extras_kv.append(extras_kv_from_mapping(mapping))
    joined = joined_table.append_column("ordinal", ordinals)
    joined = joined.append_column(
        "extras_kv",
        pa.array(extras_kv, type=pa.map_(pa.string(), pa.string())),
    )
    joined = append_constant_columns(
        joined,
        {"edge_kind": "RESOLVES_TO", "edge_layer": "SYMBOL", "extras": None},
    )
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
            "extras",
            "extras_kv",
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


def _symbol_goid_joined_table(
    symbol_goid: pa.Table,
    scip_symbols: pa.Table,
    goids: pa.Table,
) -> pa.Table:
    goid_rows, symbol_anchors, goid_anchors = _prepare_symbol_goid_inputs(
        symbol_goid=symbol_goid,
        scip_symbols=scip_symbols,
        goids=goids,
    )
    goid_project, symbol_project, goid_anchor_project = _symbol_goid_projections()
    symbol_plan = build_table_plan(
        table=symbol_anchors,
        options=TablePlanOptions(
            projection=symbol_project,
            filter_expr=E.and_(E.is_valid("repo"), E.is_valid("commit"), E.is_valid("scip_symbol")),
        ),
    )
    goid_plan = build_table_plan(
        table=goid_rows,
        options=TablePlanOptions(
            projection=goid_project,
            filter_expr=E.and_(E.is_valid("repo"), E.is_valid("commit"), E.is_valid("scip_symbol")),
        ),
    )
    joined = goid_plan.hash_join(
        right=symbol_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "scip_symbol"],
            right_keys=["repo", "commit", "scip_symbol"],
            how="left outer",
            left_output=list(goid_project.keys()),
            right_output=["src_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("src_cpg_node_id"))
    goid_anchor_plan = build_table_plan(
        table=goid_anchors,
        options=TablePlanOptions(
            projection=goid_anchor_project,
            filter_expr=E.is_valid("goid_h128"),
        ),
    )
    joined = joined.hash_join(
        right=goid_anchor_plan,
        spec=HashJoinSpec(
            left_keys=["goid_h128"],
            right_keys=["goid_h128"],
            how="left outer",
            left_output=[*goid_project.keys(), "src_cpg_node_id"],
            right_output=["dst_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("dst_cpg_node_id"))
    joined_table = _plan_to_table(joined, use_threads=True)
    if joined_table.num_rows > 0:
        joined_table = joined_table.take(
            stable_sort_indices(
                joined_table,
                sort_keys=[
                    ("repo", "ascending"),
                    ("commit", "ascending"),
                    ("scip_symbol", "ascending"),
                    ("goid_h128", "ascending"),
                ],
            )
        )
    return joined_table


def _prepare_symbol_goid_inputs(
    *,
    symbol_goid: pa.Table,
    scip_symbols: pa.Table,
    goids: pa.Table,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    goid_rows = _normalize_symbol_goid(symbol_goid)
    if "goid_h128" in goid_rows.column_names:
        goid_rows = plan_filter_or_fallback(goid_rows, is_valid_expr("goid_h128"))
    goid_rows = append_constant_columns(
        goid_rows,
        {
            "def_rel_path": None,
            "def_start_line": None,
            "def_start_col": None,
            "def_end_line": None,
            "def_end_col": None,
        },
    )
    symbol_anchors = rename_table_columns(
        _symbol_anchor_map(scip_symbols),
        {"symbol": "scip_symbol", "cpg_node_id": "src_cpg_node_id"},
    )
    goid_anchors = rename_table_columns(
        _goid_anchor_map(goids),
        {"cpg_node_id": "dst_cpg_node_id"},
    )
    goid_rows = _normalize_symbol_goid_inputs(
        goid_rows,
        table_key="core.scip_symbol_goid_xref",
        join_keys=["repo", "commit", "scip_symbol", "goid_h128"],
    )
    symbol_anchors = _normalize_symbol_goid_inputs(
        symbol_anchors,
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        join_keys=["repo", "commit", "scip_symbol"],
    )
    goid_anchors = _normalize_symbol_goid_inputs(
        goid_anchors,
        table_key=GOIDS_TABLE_KEY,
        join_keys=["goid_h128"],
    )
    return goid_rows, symbol_anchors, goid_anchors


def _normalize_symbol_goid_inputs(
    table: pa.Table,
    *,
    table_key: str,
    join_keys: list[str],
) -> pa.Table:
    allowlist = _join_safe_allowlist(table_key)
    normalized = join_safe_projection(
        normalize_table_for_join(table, allowed_columns=allowlist),
        allowed_columns=allowlist,
    )
    precheck = finalize_join_keys(
        normalized,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        precheck,
        table_key=table_key,
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    return precheck.good


def _symbol_goid_projections() -> tuple[
    dict[str, Expression], dict[str, Expression], dict[str, Expression]
]:
    goid_project = {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        "scip_symbol": E.cast(E.field("scip_symbol"), "string"),
        "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
        "def_rel_path": E.field("def_rel_path"),
        "def_start_line": E.field("def_start_line"),
        "def_start_col": E.field("def_start_col"),
        "def_end_line": E.field("def_end_line"),
        "def_end_col": E.field("def_end_col"),
    }
    symbol_project = {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        "scip_symbol": E.cast(E.field("scip_symbol"), "string"),
        "src_cpg_node_id": E.field("src_cpg_node_id"),
    }
    goid_anchor_project = {
        "goid_h128": E.cast(E.field("goid_h128"), "decimal128(38,0)"),
        "dst_cpg_node_id": E.field("dst_cpg_node_id"),
    }
    return goid_project, symbol_project, goid_anchor_project


def _symbol_anchor_map(scip_symbols: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(scip_symbols, table_key=SCIP_SYMBOLS_TABLE_KEY)
    return build_anchor_map(
        normalized,
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        pk_columns=identity_keys(SCIP_SYMBOLS_TABLE_KEY),
        include_source_pk_json=False,
    )


def _external_symbol_anchor_map(scip_symbols: pa.Table) -> pa.Table:
    normalized = canonicalize_for_table(scip_symbols, table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY)
    return build_anchor_map(
        normalized,
        table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        pk_columns=identity_keys(SCIP_EXTERNAL_SYMBOLS_TABLE_KEY),
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

    expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
    return plan_filter_or_fallback(table, expr)


def _coalesce_column(table: pa.Table, column: str, fallback: str) -> pa.Table:
    if column not in table.column_names or fallback not in table.column_names:
        return table
    values = array_from_compute("coalesce", [table[column], table[fallback]])
    if values is None:
        return table
    index = table.schema.get_field_index(column)
    if index == -1:
        return table.append_column(column, values)
    return table.set_column(index, column, values)


def _symbol_relationship_joins(
    symbol_rels: pa.Table,
    scip_symbols: pa.Table,
    scip_external_symbols: pa.Table,
) -> pa.Table:
    rels = _normalize_symbol_rels(symbol_rels)
    internal_anchor = _symbol_anchor_map(scip_symbols)
    external_anchor = _external_symbol_anchor_map(scip_external_symbols)
    joined = _join_symbol_relationship_anchor(
        rels,
        internal_anchor,
        symbol_field="symbol",
        id_field="src_cpg_node_id",
    )
    joined = _join_symbol_relationship_anchor(
        joined,
        external_anchor,
        symbol_field="symbol",
        id_field="src_cpg_node_id",
    )
    joined = _join_symbol_relationship_anchor(
        joined,
        internal_anchor,
        symbol_field="related_symbol",
        id_field="dst_cpg_node_id",
    )
    return _join_symbol_relationship_anchor(
        joined,
        external_anchor,
        symbol_field="related_symbol",
        id_field="dst_cpg_node_id",
    )


def _join_symbol_relationship_anchor(
    left: pa.Table,
    anchor: pa.Table,
    *,
    symbol_field: str,
    id_field: str,
) -> pa.Table:
    if left.num_rows == 0 or anchor.num_rows == 0:
        return left
    renamed = anchor
    if symbol_field != "symbol":
        renamed = rename_table_columns(renamed, {"symbol": symbol_field})
    if id_field != "cpg_node_id":
        renamed = rename_table_columns(renamed, {"cpg_node_id": id_field})
    left = normalize_table_for_join(left)
    renamed = normalize_table_for_join(renamed)
    left_project: dict[str, Expression] = {}
    for name in left.column_names:
        if name in {"repo", "commit", symbol_field}:
            left_project[name] = E.cast(E.field(name), "string")
        else:
            left_project[name] = E.field(name)
    right_project = {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        symbol_field: E.cast(E.field(symbol_field), "string"),
        id_field: E.field(id_field),
    }
    left_plan = build_table_plan(
        table=left,
        options=TablePlanOptions(
            projection=left_project,
            filter_expr=E.and_(E.is_valid("repo"), E.is_valid("commit"), E.is_valid(symbol_field)),
        ),
    )
    right_plan = build_table_plan(
        table=renamed,
        options=TablePlanOptions(
            projection=right_project,
            filter_expr=E.and_(E.is_valid("repo"), E.is_valid("commit"), E.is_valid(symbol_field)),
        ),
    )
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", symbol_field],
            right_keys=["repo", "commit", symbol_field],
            how="left outer",
            left_output=list(left_project.keys()),
            right_output=[id_field],
        ),
    )
    return _plan_to_table(joined, use_threads=True)


def _upsert_column(
    table: pa.Table,
    name: str,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index == -1:
        return table.append_column(name, values)
    return table.set_column(index, name, values)


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


__all__ = [
    "SymbolGoidDiagnostics",
    "SymbolRelationshipDiagnostics",
    "cpg2_edges__scip_symbol_goid_xref",
    "cpg2_edges__scip_symbol_relationships",
]
