"""CPG edge assembly exports."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.cpg import _legacy

CPG_EDGES_TABLE_KEY = _legacy.CPG_EDGES_TABLE_KEY

cpg_edge_symbol_inputs = _legacy.cpg_edge_symbol_inputs
cpg_edge_flow_inputs = _legacy.cpg_edge_flow_inputs
cpg_edge_link_inputs = _legacy.cpg_edge_link_inputs
cpg_edge_call_wiring_inputs = _legacy.cpg_edge_call_wiring_inputs
cpg_edge_syntax_node_inputs = _legacy.cpg_edge_syntax_node_inputs
cpg_edge_overlay_scope_inputs = _legacy.cpg_edge_overlay_scope_inputs
cpg_edge_overlay_symbol_inputs = _legacy.cpg_edge_overlay_symbol_inputs
cpg_edge_overlay_bytecode_inputs = _legacy.cpg_edge_overlay_bytecode_inputs
cpg_edge_overlay_syntax_call_inputs = _legacy.cpg_edge_overlay_syntax_call_inputs
cpg_edge_overlay_inspect_core_inputs = _legacy.cpg_edge_overlay_inspect_core_inputs
cpg_edge_overlay_inspect_runtime_inputs = _legacy.cpg_edge_overlay_inspect_runtime_inputs
cpg_edge_overlay_inspect_inputs = _legacy.cpg_edge_overlay_inspect_inputs
cpg_edge_overlay_inputs = _legacy.cpg_edge_overlay_inputs
cpg_edge_core_inputs = _legacy.cpg_edge_core_inputs
cpg_edges = _legacy.cpg_edges

__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "cpg_edge_call_wiring_inputs",
    "cpg_edge_core_inputs",
    "cpg_edge_flow_inputs",
    "cpg_edge_link_inputs",
    "cpg_edge_overlay_bytecode_inputs",
    "cpg_edge_overlay_inputs",
    "cpg_edge_overlay_inspect_core_inputs",
    "cpg_edge_overlay_inspect_inputs",
    "cpg_edge_overlay_inspect_runtime_inputs",
    "cpg_edge_overlay_scope_inputs",
    "cpg_edge_overlay_symbol_inputs",
    "cpg_edge_overlay_syntax_call_inputs",
    "cpg_edge_symbol_inputs",
    "cpg_edge_syntax_node_inputs",
    "cpg_edges",
]
