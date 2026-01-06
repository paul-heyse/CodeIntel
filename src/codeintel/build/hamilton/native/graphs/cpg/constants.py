"""Shared table keys and target names for CPG assembly."""

from __future__ import annotations

CPG_TARGET_NAME = "cpg"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"

SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_CALL_ARGS_TABLE_KEY = "core.syntax_call_args"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"
AST_NODES_TABLE_KEY = "core.ast_nodes"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_SCOPE_EDGES_TABLE_KEY = "core.py_sym_scope_edges"
PY_SYM_NAMESPACE_EDGES_TABLE_KEY = "core.py_sym_namespace_edges"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"
PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"
PY_BC_CFG_EDGES_TABLE_KEY = "core.py_bc_cfg_edges"
PY_BC_DEFUSE_EVENTS_TABLE_KEY = "core.py_bc_defuse_events"
PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_CLASS_MRO_TABLE_KEY = "core.py_inspect_class_mro"
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = "core.py_inspect_class_attrs"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = "core.py_inspect_runtime_state"

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
]
