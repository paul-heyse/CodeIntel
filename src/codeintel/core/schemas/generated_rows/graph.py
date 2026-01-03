"""Generated row models for insert helpers."""

from __future__ import annotations

from typing import TypedDict

__all__ = [
    "GraphCallGraphEdgesRow",
    "GraphCallGraphNodesRow",
    "GraphCdgEdgesRow",
    "GraphCfgBlocksRow",
    "GraphCfgEdgesRow",
    "GraphCpgCallTargetsRow",
    "GraphCpgEdgesArgToParamRow",
    "GraphCpgEdgesCallsRow",
    "GraphCpgEdgesRow",
    "GraphCpgNodesRow",
    "GraphDfgEdgesRow",
    "GraphImportGraphEdgesRow",
    "GraphImportModulesRow",
    "GraphPdgEdgesRow",
    "GraphSymbolUseEdgesRow",
]


class GraphCallGraphEdgesRow(TypedDict):
    """Row model for graph.call_graph_edges."""

    repo: str
    commit: str
    caller_goid_h128: int
    callee_goid_h128: int | None
    callsite_path: str
    callsite_line: int
    callsite_col: int
    language: str
    kind: str
    resolved_via: str | None
    confidence: float | None
    evidence_json: bytes | None


class GraphCallGraphNodesRow(TypedDict):
    """Row model for graph.call_graph_nodes."""

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str


class GraphCfgBlocksRow(TypedDict):
    """Row model for graph.cfg_blocks."""

    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: bytes
    in_degree: int
    out_degree: int


class GraphCfgEdgesRow(TypedDict):
    """Row model for graph.cfg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None


class GraphCdgEdgesRow(TypedDict):
    """Row model for graph.cdg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    via_succ_block_id: str
    edge_kind: str
    via_edge_kind: str | None


class GraphCpgCallTargetsRow(TypedDict):
    """Row model for graph.cpg_call_targets."""

    repo: str
    commit: str
    rel_path: str
    call_id: str
    call_node_id: str | None
    callee_symbol: str | None
    callee_def_id: str | None
    callee_def_node_id: str | None
    callee_goid_h128: int | None
    callee_entry_block_id: str | None
    callee_exit_block_id: str | None
    target_role: str
    binding_kind: str
    origin: str
    call_kind: str | None
    augop: str | None
    resolution_kind: str
    confidence: float | None
    candidate_count: int | None
    extras_json: bytes | None


class GraphCpgEdgesCallsRow(TypedDict):
    """Row model for graph.cpg_edges_calls."""

    repo: str
    commit: str
    call_id: str
    call_node_id: str | None
    callee_entry_block_id: str | None
    edge_kind: str
    confidence: float | None
    extras_json: bytes | None


class GraphCpgNodesRow(TypedDict):
    """Row model for graph.cpg_nodes."""

    repo: str
    commit: str
    cpg_node_id: int
    node_kind: str
    source_table_key: str
    source_pk_json: bytes
    rel_path: str | None
    start_byte: int | None
    end_byte: int | None
    extras_json: bytes | None


class GraphCpgEdgesRow(TypedDict):
    """Row model for graph.cpg_edges."""

    repo: str
    commit: str
    src_cpg_node_id: int
    dst_cpg_node_id: int
    edge_kind: str
    edge_layer: str
    rel_path: str | None
    ordinal: int
    extras_json: bytes | None


class GraphDfgEdgesRow(TypedDict):
    """Row model for graph.dfg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool | None
    use_kind: str | None


class GraphPdgEdgesRow(TypedDict):
    """Row model for graph.pdg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str
    src_var: str | None
    dst_var: str | None
    via_phi: bool | None
    use_kind: str | None
    via_succ_block_id: str | None
    via_edge_kind: str | None


class GraphImportGraphEdgesRow(TypedDict):
    """Row model for graph.import_graph_edges."""

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None


class GraphImportModulesRow(TypedDict):
    """Row model for graph.import_modules."""

    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int


class GraphCpgEdgesArgToParamRow(TypedDict):
    """Row model for graph.cpg_edges_arg_to_param."""

    repo: str
    commit: str
    call_id: str
    src_arg_node_id: str | None
    dst_param_node_id: str | None
    edge_kind: str
    arg_ordinal: int | None
    param_ordinal: int | None
    arg_name: str | None
    param_name: str | None
    arg_slot: str | None
    arg_role: str | None
    arg_is_implicit: bool | None
    call_kind: str | None
    augop: str | None
    confidence: float | None
    extras_json: bytes | None


class GraphCpgEdgesRetToCallRow(TypedDict):
    """Row model for graph.cpg_edges_ret_to_call."""

    repo: str
    commit: str
    call_id: str
    exit_block_id: str | None
    call_node_id: str | None
    target_role: str | None
    call_kind: str | None
    origin: str | None
    edge_kind: str
    confidence: float | None
    extras_json: bytes | None


class GraphSymbolUseEdgesRow(TypedDict):
    """Row model for graph.symbol_use_edges."""

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None
