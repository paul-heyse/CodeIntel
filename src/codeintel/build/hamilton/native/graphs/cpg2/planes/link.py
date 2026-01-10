"""Link plane CPG nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pyarrow as pa

from codeintel.build.graphs.assembly import rename_table_columns, select_table_columns
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
from codeintel.build.tabular.compute_masks import is_valid_expr
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.build.tabular.finalize_ops import (
    finalize_join_keys,
    finalize_reader,
    finalize_spec_for_table,
    record_join_precheck_errors,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan, join_safe_projection
from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.schemas.primitives import resolve_join_safe_columns

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
GOIDS_TABLE_KEY = "core.goids"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


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
    normalized = _join_ready(normalized, table_key=IMPORT_MODULES_TABLE_KEY)
    join_keys = ["repo", "commit", "module"]
    precheck = finalize_join_keys(
        normalized,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        precheck,
        table_key=IMPORT_MODULES_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=join_keys,
    )
    normalized = precheck.good
    anchors = build_anchor_map(
        normalized,
        table_key=IMPORT_MODULES_TABLE_KEY,
        pk_columns=identity_keys(IMPORT_MODULES_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = _join_ready(anchors, table_key=IMPORT_MODULES_TABLE_KEY)
    left_plan = build_table_plan(
        table=normalized,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "module": E.cast(E.field("module"), "string"),
            }
        ),
    )
    right_plan = build_table_plan(
        table=anchors,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "module": E.cast(E.field("module"), "string"),
                "cpg_node_id": E.field("cpg_node_id"),
                "source_pk_json": E.field("source_pk_json"),
            }
        ),
    )
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "module"],
            right_keys=["repo", "commit", "module"],
            how="left outer",
            left_output=["repo", "commit", "module"],
            right_output=["cpg_node_id", "source_pk_json"],
        ),
    )
    joined = joined.filter(E.is_valid("cpg_node_id"))
    joined = joined.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("module", "ascending"),
        ]
    )
    joined_table = _plan_to_table(joined, use_threads=True)
    joined = append_constant_columns(
        joined_table,
        {
            "node_kind": "MODULE",
            "source_table_key": IMPORT_MODULES_TABLE_KEY,
            "rel_path": None,
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
    joined_table = _call_graph_joined_table(call_edges, goids)
    ordinals = cpg_edge_ordinals(
        joined_table,
        table_key="graph.call_graph_edges",
        columns=[
            "caller_goid_h128",
            "callee_goid_h128",
            "callsite_path",
            "callsite_line",
            "callsite_col",
        ],
    )
    extras_fields = [
        field
        for field in ["resolved_via", "confidence", "kind"]
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
        {
            "edge_kind": "CALLS",
            "edge_layer": "FLOW",
            "extras": None,
        },
    )
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
            "extras",
            "extras_kv",
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
    joined_table = _import_graph_joined_table(import_edges, import_modules)
    ordinals = cpg_edge_ordinals(
        joined_table,
        table_key="graph.import_graph_edges",
        columns=["src_module", "dst_module", "cycle_group"],
    )
    extras_fields = [
        field
        for field in ["src_fan_out", "dst_fan_in", "cycle_group", "module_layer"]
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
        {"edge_kind": "IMPORTS", "edge_layer": "SYMBOL", "extras": None},
    )
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
            "extras",
            "extras_kv",
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


def _call_graph_joined_table(call_edges: pa.Table, goids: pa.Table) -> pa.Table:
    normalized_edges = canonicalize_for_table(call_edges, table_key="graph.call_graph_edges")
    normalized_edges = append_constant_columns(
        normalized_edges,
        {
            "callsite_line": None,
            "callsite_col": None,
            "resolved_via": None,
            "confidence": None,
            "kind": None,
        },
    )
    normalized_edges = _join_ready(normalized_edges, table_key="graph.call_graph_edges")
    edge_keys = ["caller_goid_h128", "callee_goid_h128"]
    edge_precheck = finalize_join_keys(
        normalized_edges,
        required_non_null=edge_keys,
        key_fields=edge_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        edge_precheck,
        table_key="graph.call_graph_edges",
        target_name=CPG_TARGET_NAME,
        join_keys=edge_keys,
    )
    normalized_edges = edge_precheck.good
    anchor_base = build_anchor_map(
        canonicalize_for_table(goids, table_key=GOIDS_TABLE_KEY),
        table_key=GOIDS_TABLE_KEY,
        pk_columns=identity_keys(GOIDS_TABLE_KEY),
        include_source_pk_json=False,
    )
    anchor_precheck = finalize_join_keys(
        anchor_base,
        required_non_null=["goid_h128"],
        key_fields=["goid_h128"],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        anchor_precheck,
        table_key=GOIDS_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=["goid_h128"],
    )
    anchor_base = anchor_precheck.good
    anchor_base = _join_ready(anchor_base, table_key=GOIDS_TABLE_KEY)
    src_anchor = rename_table_columns(
        anchor_base,
        {"goid_h128": "caller_goid_h128", "cpg_node_id": "src_cpg_node_id"},
    )
    src_anchor = _join_ready(src_anchor, table_key=GOIDS_TABLE_KEY)
    dst_anchor = rename_table_columns(
        anchor_base,
        {"goid_h128": "callee_goid_h128", "cpg_node_id": "dst_cpg_node_id"},
    )
    dst_anchor = _join_ready(dst_anchor, table_key=GOIDS_TABLE_KEY)
    edge_project = {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        "caller_goid_h128": E.cast(E.field("caller_goid_h128"), "decimal128(38,0)"),
        "callee_goid_h128": E.cast(E.field("callee_goid_h128"), "decimal128(38,0)"),
        "callsite_path": E.field("callsite_path"),
        "callsite_line": E.field("callsite_line"),
        "callsite_col": E.field("callsite_col"),
        "resolved_via": E.field("resolved_via"),
        "confidence": E.field("confidence"),
        "kind": E.field("kind"),
    }
    edge_plan = build_table_plan(
        table=normalized_edges,
        options=TablePlanOptions(projection=edge_project),
    )
    src_plan = build_table_plan(
        table=src_anchor,
        options=TablePlanOptions(
            projection={
                "caller_goid_h128": E.cast(E.field("caller_goid_h128"), "decimal128(38,0)"),
                "src_cpg_node_id": E.field("src_cpg_node_id"),
            }
        ),
    )
    joined = edge_plan.hash_join(
        right=src_plan,
        spec=HashJoinSpec(
            left_keys=["caller_goid_h128"],
            right_keys=["caller_goid_h128"],
            how="left outer",
            left_output=list(edge_project.keys()),
            right_output=["src_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("src_cpg_node_id"))
    dst_plan = build_table_plan(
        table=dst_anchor,
        options=TablePlanOptions(
            projection={
                "callee_goid_h128": E.cast(E.field("callee_goid_h128"), "decimal128(38,0)"),
                "dst_cpg_node_id": E.field("dst_cpg_node_id"),
            }
        ),
    )
    joined = joined.hash_join(
        right=dst_plan,
        spec=HashJoinSpec(
            left_keys=["callee_goid_h128"],
            right_keys=["callee_goid_h128"],
            how="left outer",
            left_output=[*edge_project.keys(), "src_cpg_node_id"],
            right_output=["dst_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("dst_cpg_node_id"))
    sort_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
        ("repo", "ascending"),
        ("commit", "ascending"),
        ("caller_goid_h128", "ascending"),
        ("callee_goid_h128", "ascending"),
        ("callsite_path", "ascending"),
        ("callsite_line", "ascending"),
        ("callsite_col", "ascending"),
    ]
    joined = joined.order_by(sort_keys=sort_keys)
    return _plan_to_table(joined, use_threads=True)


def _import_graph_joined_table(import_edges: pa.Table, import_modules: pa.Table) -> pa.Table:
    normalized_edges = canonicalize_for_table(import_edges, table_key="graph.import_graph_edges")
    normalized_edges = append_constant_columns(
        normalized_edges,
        {
            "src_fan_out": None,
            "dst_fan_in": None,
            "cycle_group": None,
            "module_layer": None,
        },
    )
    normalized_edges = _join_ready(normalized_edges, table_key="graph.import_graph_edges")
    edge_keys = ["repo", "commit", "src_module", "dst_module"]
    edge_precheck = finalize_join_keys(
        normalized_edges,
        required_non_null=edge_keys,
        key_fields=edge_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        edge_precheck,
        table_key="graph.import_graph_edges",
        target_name=CPG_TARGET_NAME,
        join_keys=edge_keys,
    )
    normalized_edges = edge_precheck.good
    anchor_base = build_anchor_map(
        canonicalize_for_table(
            select_table_columns(import_modules, ["repo", "commit", "module"]),
            table_key=IMPORT_MODULES_TABLE_KEY,
        ),
        table_key=IMPORT_MODULES_TABLE_KEY,
        pk_columns=identity_keys(IMPORT_MODULES_TABLE_KEY),
        include_source_pk_json=False,
    )
    anchor_precheck = finalize_join_keys(
        anchor_base,
        required_non_null=["repo", "commit", "module"],
        key_fields=["repo", "commit", "module"],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        anchor_precheck,
        table_key=IMPORT_MODULES_TABLE_KEY,
        target_name=CPG_TARGET_NAME,
        join_keys=["repo", "commit", "module"],
    )
    anchor_base = anchor_precheck.good
    anchor_base = _join_ready(anchor_base, table_key=IMPORT_MODULES_TABLE_KEY)
    src_anchor = rename_table_columns(
        anchor_base,
        {"module": "src_module", "cpg_node_id": "src_cpg_node_id"},
    )
    src_anchor = _join_ready(src_anchor, table_key=IMPORT_MODULES_TABLE_KEY)
    dst_anchor = rename_table_columns(
        anchor_base,
        {"module": "dst_module", "cpg_node_id": "dst_cpg_node_id"},
    )
    dst_anchor = _join_ready(dst_anchor, table_key=IMPORT_MODULES_TABLE_KEY)
    edge_project = {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        "src_module": E.cast(E.field("src_module"), "string"),
        "dst_module": E.cast(E.field("dst_module"), "string"),
        "src_fan_out": E.field("src_fan_out"),
        "dst_fan_in": E.field("dst_fan_in"),
        "cycle_group": E.field("cycle_group"),
        "module_layer": E.field("module_layer"),
    }
    edge_plan = build_table_plan(
        table=normalized_edges,
        options=TablePlanOptions(projection=edge_project),
    )
    src_plan = build_table_plan(
        table=src_anchor,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "src_module": E.cast(E.field("src_module"), "string"),
                "src_cpg_node_id": E.field("src_cpg_node_id"),
            }
        ),
    )
    joined = edge_plan.hash_join(
        right=src_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "src_module"],
            right_keys=["repo", "commit", "src_module"],
            how="left outer",
            left_output=list(edge_project.keys()),
            right_output=["src_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("src_cpg_node_id"))
    dst_plan = build_table_plan(
        table=dst_anchor,
        options=TablePlanOptions(
            projection={
                "repo": E.cast(E.field("repo"), "string"),
                "commit": E.cast(E.field("commit"), "string"),
                "dst_module": E.cast(E.field("dst_module"), "string"),
                "dst_cpg_node_id": E.field("dst_cpg_node_id"),
            }
        ),
    )
    joined = joined.hash_join(
        right=dst_plan,
        spec=HashJoinSpec(
            left_keys=["repo", "commit", "dst_module"],
            right_keys=["repo", "commit", "dst_module"],
            how="left outer",
            left_output=[*edge_project.keys(), "src_cpg_node_id"],
            right_output=["dst_cpg_node_id"],
        ),
    )
    joined = joined.filter(E.is_valid("dst_cpg_node_id"))
    sort_keys: list[tuple[str, Literal["ascending", "descending"]]] = [
        ("repo", "ascending"),
        ("commit", "ascending"),
        ("src_module", "ascending"),
        ("dst_module", "ascending"),
        ("cycle_group", "ascending"),
    ]
    joined = joined.order_by(sort_keys=sort_keys)
    return _plan_to_table(joined, use_threads=True)


def _plan_to_table(plan: Plan, *, use_threads: bool) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    if not use_threads:
        execution_ctx = replace(execution_ctx, use_threads=False)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            _INTERNAL_PLAN_TABLE_KEY,
            mode="tolerant",
            ordering=plan.ordering,
        ),
    )
    return result.good


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
    "CallGraphDiagnostics",
    "ImportGraphDiagnostics",
    "ImportModuleDiagnostics",
    "cpg2_edges__call_graph_edges",
    "cpg2_edges__import_graph_edges",
    "cpg2_nodes__import_modules",
]
