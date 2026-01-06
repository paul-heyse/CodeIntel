"""AST plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.serialization.payload import encode_payload

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
AST_NODES_TABLE_KEY = "core.ast_nodes"


def cpg2_nodes__ast_nodes(ast_nodes: pa.Table, env: BuildEnv) -> pa.Table:
    """Build CPG nodes from AST nodes.

    Returns
    -------
    pyarrow.Table
        CPG node table for AST nodes.
    """
    required = {"path", "node_type", "hash", "start_byte", "end_byte"}
    if not required.issubset(set(ast_nodes.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(ast_nodes):
        pk_values = pk_from_row(row, table_key=AST_NODES_TABLE_KEY)
        extras_values = {
            "node_type": row.get("node_type"),
            "name": row.get("name"),
            "qualname": row.get("qualname"),
            "parent_qualname": row.get("parent_qualname"),
            "lineno": row.get("lineno"),
            "end_lineno": row.get("end_lineno"),
            "col_offset": row.get("col_offset"),
            "end_col_offset": row.get("end_col_offset"),
            "decorator_start_line": row.get("decorator_start_line"),
            "decorator_end_line": row.get("decorator_end_line"),
            "decorators": row.get("decorators"),
            "docstring": row.get("docstring"),
            "ctx": row.get("ctx"),
            "type_comment": row.get("type_comment"),
            "type_ignores": row.get("type_ignores"),
            "identifier": row.get("identifier"),
            "attribute": row.get("attribute"),
            "imported": row.get("imported"),
            "asname": row.get("asname"),
            "module": row.get("module"),
            "level": row.get("level"),
            "constant_kind": row.get("constant_kind"),
        }
        rows.append(
            {
                "repo": env.repo,
                "commit": env.commit,
                "cpg_node_id": cpg_node_id(AST_NODES_TABLE_KEY, pk_values),
                "node_kind": "AST_NODE",
                "source_table_key": AST_NODES_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
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


__all__ = ["cpg2_nodes__ast_nodes"]
