"""Python symtable plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY = "core.py_sym_unresolved_bindings"


def cpg2_nodes__py_sym_scopes(scopes: pa.Table) -> pa.Table:
    """Build CPG nodes from symtable scopes.

    Returns
    -------
    pyarrow.Table
        CPG node table for symtable scopes.
    """
    required = {"repo", "commit", "rel_path", "scope_id", "scope_type"}
    if not required.issubset(set(scopes.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(scopes):
        pk_values = pk_from_row(row, table_key=PY_SYM_SCOPES_TABLE_KEY)
        extras_values = {
            "scope_type": row.get("scope_type"),
            "scope_name": row.get("scope_name"),
            "qualpath": row.get("qualpath"),
            "lineno": row.get("lineno"),
            "is_nested": row.get("is_nested"),
            "is_optimized": row.get("is_optimized"),
            "has_children": row.get("has_children"),
            "parent_scope_id": row.get("parent_scope_id"),
            "anchor_ast_node_id": row.get("anchor_ast_node_id"),
            "anchor_confidence": row.get("anchor_confidence"),
            "anchor_reason": row.get("anchor_reason"),
            "scope_local_id": row.get("scope_local_id"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, pk_values),
                "node_kind": "SCOPE",
                "source_table_key": PY_SYM_SCOPES_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_sym_bindings(bindings: pa.Table) -> pa.Table:
    """Build CPG nodes from symtable bindings.

    Returns
    -------
    pyarrow.Table
        CPG node table for symtable bindings.
    """
    required = {"repo", "commit", "rel_path", "binding_id", "scope_id", "name", "binding_kind"}
    if not required.issubset(set(bindings.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(bindings):
        pk_values = pk_from_row(row, table_key=PY_SYM_BINDINGS_TABLE_KEY)
        extras_values = {
            "scope_id": row.get("scope_id"),
            "name": row.get("name"),
            "binding_kind": row.get("binding_kind"),
            "declared_here": row.get("declared_here"),
            "referenced_here": row.get("referenced_here"),
            "assigned_here": row.get("assigned_here"),
            "annotated_here": row.get("annotated_here"),
            "scoping_class": row.get("scoping_class"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, pk_values),
                "node_kind": "BINDING",
                "source_table_key": PY_SYM_BINDINGS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": None,
                "end_byte": None,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_sym_unresolved_bindings(bindings: pa.Table) -> pa.Table:
    """Build CPG nodes for unresolved symtable bindings.

    Returns
    -------
    pyarrow.Table
        CPG node table for unresolved symtable bindings.
    """
    required = {"repo", "commit", "rel_path", "binding_id"}
    if not required.issubset(set(bindings.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(bindings):
        pk_values = pk_from_row(row, table_key=PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY)
        extras_values = {
            "resolution_kind": row.get("resolution_kind"),
            "confidence": row.get("confidence"),
            "reason": row.get("reason"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY, pk_values),
                "node_kind": "BINDING_UNRESOLVED",
                "source_table_key": PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": None,
                "end_byte": None,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


__all__ = [
    "cpg2_nodes__py_sym_bindings",
    "cpg2_nodes__py_sym_scopes",
    "cpg2_nodes__py_sym_unresolved_bindings",
]
