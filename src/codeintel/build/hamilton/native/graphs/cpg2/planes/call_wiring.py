"""Call wiring plane CPG edges."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    canonicalize_for_table,
    lookup_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.core.columnar.rows import table_for_rows

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"


@dataclass(frozen=True)
class CallWiringDiagnostics:
    """Diagnostics for call wiring edge resolution."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


def cpg2_edges__call_wiring_calls(
    call_edges: pa.Table,
    cfg_blocks: pa.Table,
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for call wiring (callsite -> callee entry).

    Returns
    -------
    pyarrow.Table
        CPG edges for call wiring.
    """
    syntax_index = _syntax_node_index(syntax_nodes)
    block_index = _block_id_index(cfg_blocks)
    rows: list[dict[str, object]] = []
    for row in table_rows(call_edges):
        call_node_id = row.get("call_node_id")
        if call_node_id is None:
            continue
        syntax_info = syntax_index.get(
            _syntax_lookup_key(row.get("repo"), row.get("commit"), call_node_id)
        )
        if syntax_info is None:
            continue
        block_info = block_index.get(row.get("callee_entry_block_id"))
        if block_info is None:
            continue
        block_idx = block_info.get("block_idx")
        function_goid = block_info.get("function_goid_h128")
        if block_idx is None or function_goid is None:
            continue
        syntax_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": syntax_info.get("rel_path"),
            "producer": syntax_info.get("producer"),
            "node_id": call_node_id,
        }
        cfg_pk = {
            "function_goid_h128": function_goid,
            "block_idx": block_idx,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "confidence": row.get("confidence"),
            "call_extras": row.get("extras_kv"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_calls",
            {
                "call_id": row.get("call_id"),
                "callee_entry_block_id": row.get("callee_entry_block_id"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, syntax_pk),
                "dst_cpg_node_id": cpg_node_id(CFG_BLOCKS_TABLE_KEY, cfg_pk),
                "edge_kind": row.get("edge_kind") or "CALLS",
                "edge_layer": "FLOW",
                "rel_path": syntax_info.get("rel_path"),
                "ordinal": ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, rows)
    _record_diagnostics(diagnostics, "call_wiring_calls", call_edges, row_count)
    return table


def cpg2_edges__call_wiring_arg_to_param(
    arg_edges: pa.Table,
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for argument-to-parameter wiring.

    Returns
    -------
    pyarrow.Table
        CPG edges for arg->param wiring.
    """
    syntax_index = _syntax_node_index(syntax_nodes)
    rows: list[dict[str, object]] = []
    for row in table_rows(arg_edges):
        src_arg_node_id = row.get("src_arg_node_id")
        dst_param_node_id = row.get("dst_param_node_id")
        if src_arg_node_id is None or dst_param_node_id is None:
            continue
        src_info = syntax_index.get(
            _syntax_lookup_key(row.get("repo"), row.get("commit"), src_arg_node_id)
        )
        dst_info = syntax_index.get(
            _syntax_lookup_key(row.get("repo"), row.get("commit"), dst_param_node_id)
        )
        if src_info is None or dst_info is None:
            continue
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": src_info.get("rel_path"),
            "producer": src_info.get("producer"),
            "node_id": src_arg_node_id,
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": dst_info.get("rel_path"),
            "producer": dst_info.get("producer"),
            "node_id": dst_param_node_id,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "arg_ordinal": row.get("arg_ordinal"),
            "param_ordinal": row.get("param_ordinal"),
            "arg_name": row.get("arg_name"),
            "param_name": row.get("param_name"),
            "arg_slot": row.get("arg_slot"),
            "arg_role": row.get("arg_role"),
            "arg_is_implicit": row.get("arg_is_implicit"),
            "call_kind": row.get("call_kind"),
            "augop": row.get("augop"),
            "confidence": row.get("confidence"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_arg_to_param",
            {
                "call_id": row.get("call_id"),
                "arg_ordinal": row.get("arg_ordinal"),
                "param_ordinal": row.get("param_ordinal"),
                "src_arg_node_id": src_arg_node_id,
                "dst_param_node_id": dst_param_node_id,
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, src_pk),
                "dst_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, dst_pk),
                "edge_kind": row.get("edge_kind") or "ARG_TO_PARAM",
                "edge_layer": "FLOW",
                "rel_path": src_info.get("rel_path"),
                "ordinal": ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, rows)
    _record_diagnostics(diagnostics, "call_wiring_args", arg_edges, row_count)
    return table


def cpg2_edges__call_wiring_ret_to_call(
    ret_edges: pa.Table,
    cfg_blocks: pa.Table,
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges for return-to-call wiring.

    Returns
    -------
    pyarrow.Table
        CPG edges for return wiring.
    """
    syntax_index = _syntax_node_index(syntax_nodes)
    block_index = _block_id_index(cfg_blocks)
    rows: list[dict[str, object]] = []
    for row in table_rows(ret_edges):
        call_node_id = row.get("call_node_id")
        if call_node_id is None:
            continue
        syntax_info = syntax_index.get(
            _syntax_lookup_key(row.get("repo"), row.get("commit"), call_node_id)
        )
        if syntax_info is None:
            continue
        block_info = block_index.get(row.get("exit_block_id"))
        if block_info is None:
            continue
        function_goid = block_info.get("function_goid_h128")
        block_idx = block_info.get("block_idx")
        if function_goid is None or block_idx is None:
            continue
        cfg_pk = {
            "function_goid_h128": function_goid,
            "block_idx": block_idx,
        }
        syntax_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": syntax_info.get("rel_path"),
            "producer": syntax_info.get("producer"),
            "node_id": call_node_id,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "confidence": row.get("confidence"),
            "target_role": row.get("target_role"),
            "call_kind": row.get("call_kind"),
            "origin": row.get("origin"),
            "summary": row.get("extras_kv"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_ret_to_call",
            {"call_id": row.get("call_id"), "exit_block_id": row.get("exit_block_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(CFG_BLOCKS_TABLE_KEY, cfg_pk),
                "dst_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, syntax_pk),
                "edge_kind": row.get("edge_kind") or "RET_TO_CALL",
                "edge_layer": "FLOW",
                "rel_path": syntax_info.get("rel_path"),
                "ordinal": ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, rows)
    _record_diagnostics(diagnostics, "call_wiring_return", ret_edges, row_count)
    return table


def _syntax_node_index(
    syntax_nodes: pa.Table,
) -> dict[tuple[object, ...], dict[str, object]]:
    index: dict[tuple[object, ...], dict[str, object]] = {}
    normalized = canonicalize_for_table(syntax_nodes, table_key=SYNTAX_NODES_TABLE_KEY)
    key_columns = lookup_keys(SYNTAX_NODES_TABLE_KEY, "node_id")
    for row in table_rows(normalized):
        key = tuple(row.get(column) for column in key_columns)
        index[key] = {
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
        }
    return index


def _block_id_index(cfg_blocks: pa.Table) -> dict[object, dict[str, object]]:
    index: dict[object, dict[str, object]] = {}
    normalized = canonicalize_for_table(cfg_blocks, table_key=CFG_BLOCKS_TABLE_KEY)
    block_key = lookup_keys(CFG_BLOCKS_TABLE_KEY, "block_id")
    block_id_index = block_key[0]
    for row in table_rows(normalized):
        block_id = row.get(block_id_index)
        if block_id is None:
            continue
        index[block_id] = {
            "function_goid_h128": row.get("function_goid_h128"),
            "block_idx": row.get("block_idx"),
        }
    return index


def _syntax_lookup_key(
    repo: object,
    commit: object,
    node_id: object,
) -> tuple[object, ...]:
    key_columns = lookup_keys(SYNTAX_NODES_TABLE_KEY, "node_id")
    values = {"repo": repo, "commit": commit, "node_id": node_id}
    return tuple(values.get(column) for column in key_columns)


def _record_diagnostics(
    diagnostics: dict[str, object] | None,
    key: str,
    input_edges: pa.Table,
    resolved_edges: int,
) -> None:
    if diagnostics is None:
        return
    total_edges = input_edges.num_rows
    dropped_edges = max(total_edges - resolved_edges, 0)
    diagnostics[key] = CallWiringDiagnostics(
        total_edges=total_edges,
        resolved_edges=resolved_edges,
        dropped_edges=dropped_edges,
    )


__all__ = [
    "CallWiringDiagnostics",
    "cpg2_edges__call_wiring_arg_to_param",
    "cpg2_edges__call_wiring_calls",
    "cpg2_edges__call_wiring_ret_to_call",
]
