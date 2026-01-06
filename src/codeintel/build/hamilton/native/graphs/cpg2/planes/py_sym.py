"""Python symtable plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.serialization.payload import encode_payload

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"


def cpg_nodes_from_py_sym_scopes(scopes: pa.Table) -> pa.Table:
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
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg_nodes_from_py_sym_bindings(bindings: pa.Table) -> pa.Table:
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
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def _payload_bytes(values: dict[str, object]) -> bytes:
    encoded = encode_payload(values)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


__all__ = ["cpg_nodes_from_py_sym_bindings", "cpg_nodes_from_py_sym_scopes"]
