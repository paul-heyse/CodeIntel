"""Unified CPG node and edge assembly for property graph exports."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.cpg import _legacy
from codeintel.build.hamilton.native.graphs.cpg.bytecode import (
    instruction_cpg_id,
    py_bc_callsite_edges_to_cpg,
    py_bc_callsite_symbol_edges_to_cpg,
    py_bc_stack_edges_to_cpg,
)
from codeintel.build.hamilton.native.graphs.cpg.edges import (
    CPG_EDGES_TABLE_KEY,
    cpg_edge_call_wiring_inputs,
    cpg_edge_core_inputs,
    cpg_edge_flow_inputs,
    cpg_edge_link_inputs,
    cpg_edge_overlay_bytecode_inputs,
    cpg_edge_overlay_inputs,
    cpg_edge_overlay_inspect_core_inputs,
    cpg_edge_overlay_inspect_inputs,
    cpg_edge_overlay_inspect_runtime_inputs,
    cpg_edge_overlay_scope_inputs,
    cpg_edge_overlay_symbol_inputs,
    cpg_edge_overlay_syntax_call_inputs,
    cpg_edge_symbol_inputs,
    cpg_edge_syntax_node_inputs,
    cpg_edges,
)
from codeintel.build.hamilton.native.graphs.cpg.ids import stable_cpg_id
from codeintel.build.hamilton.native.graphs.cpg.inspect_overlay import (
    inspect_to_ast_edges_to_cpg,
    py_inspect_unwrap_edges_to_cpg,
)
from codeintel.build.hamilton.native.graphs.cpg.nodes import (
    CPG_NODES_TABLE_KEY,
    cpg_nodes,
    cpg_nodes__core_inputs,
    cpg_nodes__graph_inputs,
    cpg_nodes__inputs,
    cpg_nodes__inspect_inputs,
    cpg_nodes__py_inputs,
    cpg_nodes__syntax_inputs,
)

CPG_TARGET_NAME = _legacy.CPG_TARGET_NAME
SYNTAX_NODES_TABLE_KEY = _legacy.SYNTAX_NODES_TABLE_KEY
SYNTAX_CALLS_TABLE_KEY = _legacy.SYNTAX_CALLS_TABLE_KEY
SYNTAX_CALL_ARGS_TABLE_KEY = _legacy.SYNTAX_CALL_ARGS_TABLE_KEY
SCIP_SYMBOLS_TABLE_KEY = _legacy.SCIP_SYMBOLS_TABLE_KEY
GOIDS_TABLE_KEY = _legacy.GOIDS_TABLE_KEY
CFG_BLOCKS_TABLE_KEY = _legacy.CFG_BLOCKS_TABLE_KEY
IMPORT_MODULES_TABLE_KEY = _legacy.IMPORT_MODULES_TABLE_KEY
TS_TOKENS_TABLE_KEY = _legacy.TS_TOKENS_TABLE_KEY
TS_TRIVIA_TABLE_KEY = _legacy.TS_TRIVIA_TABLE_KEY
AST_NODES_TABLE_KEY = _legacy.AST_NODES_TABLE_KEY
PY_SYM_SCOPES_TABLE_KEY = _legacy.PY_SYM_SCOPES_TABLE_KEY
PY_SYM_BINDINGS_TABLE_KEY = _legacy.PY_SYM_BINDINGS_TABLE_KEY
PY_SYM_SCOPE_EDGES_TABLE_KEY = _legacy.PY_SYM_SCOPE_EDGES_TABLE_KEY
PY_SYM_NAMESPACE_EDGES_TABLE_KEY = _legacy.PY_SYM_NAMESPACE_EDGES_TABLE_KEY
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = _legacy.PY_SYM_RESOLUTION_EDGES_TABLE_KEY
PY_BC_CODE_UNITS_TABLE_KEY = _legacy.PY_BC_CODE_UNITS_TABLE_KEY
PY_BC_INSTRUCTIONS_TABLE_KEY = _legacy.PY_BC_INSTRUCTIONS_TABLE_KEY
PY_BC_BLOCKS_TABLE_KEY = _legacy.PY_BC_BLOCKS_TABLE_KEY
PY_BC_CFG_EDGES_TABLE_KEY = _legacy.PY_BC_CFG_EDGES_TABLE_KEY
PY_BC_DEFUSE_EVENTS_TABLE_KEY = _legacy.PY_BC_DEFUSE_EVENTS_TABLE_KEY
PY_INSPECT_OBJECTS_TABLE_KEY = _legacy.PY_INSPECT_OBJECTS_TABLE_KEY
PY_INSPECT_CLASS_MRO_TABLE_KEY = _legacy.PY_INSPECT_CLASS_MRO_TABLE_KEY
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = _legacy.PY_INSPECT_CLASS_ATTRS_TABLE_KEY
PY_INSPECT_UNWRAP_TABLE_KEY = _legacy.PY_INSPECT_UNWRAP_TABLE_KEY
PY_INSPECT_SIGNATURES_TABLE_KEY = _legacy.PY_INSPECT_SIGNATURES_TABLE_KEY
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = _legacy.PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY
PY_INSPECT_SOURCE_TABLE_KEY = _legacy.PY_INSPECT_SOURCE_TABLE_KEY
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = _legacy.PY_INSPECT_RUNTIME_STATE_TABLE_KEY

cpg__options = _legacy.cpg__options
cpg__overlay_options = _legacy.cpg__overlay_options

__all__ = [
    "AST_NODES_TABLE_KEY",
    "CFG_BLOCKS_TABLE_KEY",
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "CPG_TARGET_NAME",
    "GOIDS_TABLE_KEY",
    "IMPORT_MODULES_TABLE_KEY",
    "PY_BC_BLOCKS_TABLE_KEY",
    "PY_BC_CFG_EDGES_TABLE_KEY",
    "PY_BC_CODE_UNITS_TABLE_KEY",
    "PY_BC_DEFUSE_EVENTS_TABLE_KEY",
    "PY_BC_INSTRUCTIONS_TABLE_KEY",
    "PY_INSPECT_CLASS_ATTRS_TABLE_KEY",
    "PY_INSPECT_CLASS_MRO_TABLE_KEY",
    "PY_INSPECT_OBJECTS_TABLE_KEY",
    "PY_INSPECT_RUNTIME_STATE_TABLE_KEY",
    "PY_INSPECT_SIGNATURES_TABLE_KEY",
    "PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY",
    "PY_INSPECT_SOURCE_TABLE_KEY",
    "PY_INSPECT_UNWRAP_TABLE_KEY",
    "PY_SYM_BINDINGS_TABLE_KEY",
    "PY_SYM_NAMESPACE_EDGES_TABLE_KEY",
    "PY_SYM_RESOLUTION_EDGES_TABLE_KEY",
    "PY_SYM_SCOPES_TABLE_KEY",
    "PY_SYM_SCOPE_EDGES_TABLE_KEY",
    "SCIP_SYMBOLS_TABLE_KEY",
    "SYNTAX_CALLS_TABLE_KEY",
    "SYNTAX_CALL_ARGS_TABLE_KEY",
    "SYNTAX_NODES_TABLE_KEY",
    "TS_TOKENS_TABLE_KEY",
    "TS_TRIVIA_TABLE_KEY",
    "cpg__options",
    "cpg__overlay_options",
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
    "cpg_nodes",
    "cpg_nodes__core_inputs",
    "cpg_nodes__graph_inputs",
    "cpg_nodes__inputs",
    "cpg_nodes__inspect_inputs",
    "cpg_nodes__py_inputs",
    "cpg_nodes__syntax_inputs",
    "inspect_to_ast_edges_to_cpg",
    "instruction_cpg_id",
    "py_bc_callsite_edges_to_cpg",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
    "py_inspect_unwrap_edges_to_cpg",
    "stable_cpg_id",
]
