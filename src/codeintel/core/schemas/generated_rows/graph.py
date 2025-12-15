"""Generated row models for insert helpers."""

from __future__ import annotations

from typing import TypedDict

__all__ = [
    "GraphCallGraphEdgesRow",
    "GraphCallGraphNodesRow",
    "GraphCfgBlocksRow",
    "GraphCfgEdgesRow",
    "GraphDfgEdgesRow",
    "GraphImportGraphEdgesRow",
    "GraphImportModulesRow",
    "GraphSymbolUseEdgesRow",
    "GraphVCallDepthStatsRow",
    "GraphVFunctionCallCountsRow",
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
    evidence_json: object | None


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
    stmts_json: object
    in_degree: int
    out_degree: int


class GraphCfgEdgesRow(TypedDict):
    """Row model for graph.cfg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None


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


class GraphSymbolUseEdgesRow(TypedDict):
    """Row model for graph.symbol_use_edges."""

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None


class GraphVCallDepthStatsRow(TypedDict):
    """Row model for graph.v_call_depth_stats."""

    repo: str
    commit: str
    function_goid_h128: int
    max_call_depth: int
    is_leaf: bool


class GraphVFunctionCallCountsRow(TypedDict):
    """Row model for graph.v_function_call_counts."""

    repo: str
    commit: str
    function_goid_h128: int
    num_callees: int
    num_unique_callees: int
    num_callers: int
