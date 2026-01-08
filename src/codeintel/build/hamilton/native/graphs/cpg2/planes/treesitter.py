"""Tree-sitter plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping, extras_kv_from_payload
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"


def cpg2_nodes__ts_tokens(tokens: pa.Table) -> pa.Table:
    """Build CPG nodes from tree-sitter tokens.

    Returns
    -------
    pyarrow.Table
        CPG node table for tree-sitter tokens.
    """
    required = {
        "repo",
        "commit",
        "rel_path",
        "language",
        "token_id",
        "token_kind",
        "node_type",
        "start_byte",
        "end_byte",
        "text_preview",
    }
    if not required.issubset(set(tokens.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(tokens):
        pk_values = pk_from_row(row, table_key=TS_TOKENS_TABLE_KEY)
        extras_values = {
            "token_kind": row.get("token_kind"),
            "node_type": row.get("node_type"),
            "text_preview": row.get("text_preview"),
        }
        extras_kv = extras_kv_from_payload(row.get("extras_json"))
        value_kv = extras_kv_from_mapping(extras_values)
        if extras_kv is None:
            extras_kv = value_kv
        elif value_kv:
            extras_kv.update(value_kv)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(TS_TOKENS_TABLE_KEY, pk_values),
                "node_kind": "TS_TOKEN",
                "source_table_key": TS_TOKENS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__ts_trivia(trivia: pa.Table) -> pa.Table:
    """Build CPG nodes from tree-sitter trivia.

    Returns
    -------
    pyarrow.Table
        CPG node table for tree-sitter trivia.
    """
    required = {
        "repo",
        "commit",
        "rel_path",
        "language",
        "trivia_id",
        "trivia_kind",
        "node_type",
        "start_byte",
        "end_byte",
        "text_preview",
    }
    if not required.issubset(set(trivia.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(trivia):
        pk_values = pk_from_row(row, table_key=TS_TRIVIA_TABLE_KEY)
        extras_values = {
            "trivia_kind": row.get("trivia_kind"),
            "node_type": row.get("node_type"),
            "text_preview": row.get("text_preview"),
        }
        extras_kv = extras_kv_from_payload(row.get("extras_json"))
        value_kv = extras_kv_from_mapping(extras_values)
        if extras_kv is None:
            extras_kv = value_kv
        elif value_kv:
            extras_kv.update(value_kv)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(TS_TRIVIA_TABLE_KEY, pk_values),
                "node_kind": "TS_TRIVIA",
                "source_table_key": TS_TRIVIA_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


__all__ = ["cpg2_nodes__ts_tokens", "cpg2_nodes__ts_trivia"]
