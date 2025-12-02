"""Graph table TypedDict row models and serializers.

This module provides TypedDict definitions for graph DuckDB tables:
- CallGraphNodeRow for graph.call_graph_nodes
- CallGraphEdgeRow for graph.call_graph_edges
- ImportEdgeRow for graph.import_graph_edges
- ImportModuleRow for graph.import_modules
- CFGBlockRow for graph.cfg_blocks
- CFGEdgeRow for graph.cfg_edges
- DFGEdgeRow for graph.dfg_edges
- SymbolUseRow for graph.symbol_use_edges
"""

from __future__ import annotations

from typing import TypedDict


class SymbolUseRow(TypedDict):
    """Row shape for graph.symbol_use_edges inserts.

    Parameters
    ----------
    symbol
        Symbol identifier.
    def_path
        Definition file path.
    use_path
        Usage file path.
    same_file
        Definition and use in same file.
    same_module
        Definition and use in same module.
    def_goid_h128
        Definition GOID hash.
    use_goid_h128
        Usage GOID hash.
    """

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None


def symbol_use_to_tuple(row: SymbolUseRow) -> tuple[object, ...]:
    """Serialize a SymbolUseRow into the INSERT column order.

    Parameters
    ----------
    row
        The symbol use row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by symbol_use_edges INSERTs.
    """
    return (
        row["symbol"],
        row["def_path"],
        row["use_path"],
        row["same_file"],
        row["same_module"],
        row["def_goid_h128"],
        row["use_goid_h128"],
    )


class CallGraphNodeRow(TypedDict):
    """Row shape for graph.call_graph_nodes inserts.

    Parameters
    ----------
    goid_h128
        128-bit hash of the function GOID.
    language
        Programming language.
    kind
        Function kind.
    arity
        Number of parameters.
    is_public
        Whether the function is public.
    rel_path
        Relative file path.
    """

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str


def call_graph_node_to_tuple(row: CallGraphNodeRow) -> tuple[object, ...]:
    """Serialize a CallGraphNodeRow into the INSERT column order.

    Parameters
    ----------
    row
        The call graph node row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_nodes INSERT order.
    """
    return (
        row["goid_h128"],
        row["language"],
        row["kind"],
        row["arity"],
        row["is_public"],
        row["rel_path"],
    )


class CallGraphEdgeRow(TypedDict):
    """Row shape for graph.call_graph_edges inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    caller_goid_h128
        Caller function GOID hash.
    callee_goid_h128
        Callee function GOID hash.
    callsite_path
        Path to callsite file.
    callsite_line
        Callsite line number.
    callsite_col
        Callsite column number.
    language
        Programming language.
    kind
        Call kind.
    resolved_via
        Resolution method used.
    confidence
        Resolution confidence score.
    evidence_json
        Evidence for the call (JSON).
    """

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
    evidence_json: object


def call_graph_edge_to_tuple(row: CallGraphEdgeRow) -> tuple[object, ...]:
    """Serialize a CallGraphEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The call graph edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_edges INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["caller_goid_h128"],
        row["callee_goid_h128"],
        row["callsite_path"],
        row["callsite_line"],
        row["callsite_col"],
        row["language"],
        row["kind"],
        row["resolved_via"],
        row["confidence"],
        row["evidence_json"],
    )


class ImportEdgeRow(TypedDict):
    """Row shape for graph.import_graph_edges inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    src_module
        Source module.
    dst_module
        Destination module.
    src_fan_out
        Source module fan-out.
    dst_fan_in
        Destination module fan-in.
    cycle_group
        Cycle group identifier.
    module_layer
        Module layer in hierarchy.
    """

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None


def import_edge_to_tuple(row: ImportEdgeRow) -> tuple[object, ...]:
    """Serialize an ImportEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The import edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with import_graph_edges INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["src_module"],
        row["dst_module"],
        row["src_fan_out"],
        row["dst_fan_in"],
        row["cycle_group"],
        row.get("module_layer"),
    )


class ImportModuleRow(TypedDict):
    """Row shape for graph.import_modules inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    module
        Module name.
    scc_id
        Strongly connected component ID.
    component_size
        Size of the component.
    layer
        Module layer in hierarchy.
    cycle_group
        Cycle group identifier.
    """

    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int


def import_module_to_tuple(row: ImportModuleRow) -> tuple[object, ...]:
    """Serialize an ImportModuleRow into the INSERT column order.

    Parameters
    ----------
    row
        The import module row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with import_modules INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["module"],
        row["scc_id"],
        row["component_size"],
        row.get("layer"),
        row["cycle_group"],
    )


class CFGBlockRow(TypedDict):
    """Row shape for graph.cfg_blocks inserts.

    Parameters
    ----------
    function_goid_h128
        Function GOID hash.
    block_idx
        Block index.
    block_id
        Block identifier.
    label
        Block label.
    file_path
        Source file path.
    start_line
        Starting line number.
    end_line
        Ending line number.
    kind
        Block kind.
    stmts_json
        Statements in block (JSON).
    in_degree
        Incoming edge count.
    out_degree
        Outgoing edge count.
    """

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


def cfg_block_to_tuple(row: CFGBlockRow) -> tuple[object, ...]:
    """Serialize a CFGBlockRow into the INSERT column order.

    Parameters
    ----------
    row
        The CFG block row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with cfg_blocks INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["block_idx"],
        row["block_id"],
        row["label"],
        row["file_path"],
        row["start_line"],
        row["end_line"],
        row["kind"],
        row["stmts_json"],
        row["in_degree"],
        row["out_degree"],
    )


class CFGEdgeRow(TypedDict):
    """Row shape for graph.cfg_edges inserts.

    Parameters
    ----------
    function_goid_h128
        Function GOID hash.
    src_block_id
        Source block identifier.
    dst_block_id
        Destination block identifier.
    edge_kind
        Edge kind.
    """

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None


def cfg_edge_to_tuple(row: CFGEdgeRow) -> tuple[object, ...]:
    """Serialize a CFGEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The CFG edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with cfg_edges INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["src_block_id"],
        row["dst_block_id"],
        row["edge_kind"],
    )


class DFGEdgeRow(TypedDict):
    """Row shape for graph.dfg_edges inserts.

    Parameters
    ----------
    function_goid_h128
        Function GOID hash.
    src_block_id
        Source block identifier.
    dst_block_id
        Destination block identifier.
    src_var
        Source variable name.
    dst_var
        Destination variable name.
    edge_kind
        Edge kind.
    via_phi
        Whether edge goes through phi node.
    use_kind
        Kind of use (read, write, etc.).
    """

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool
    use_kind: str | None


def dfg_edge_to_tuple(row: DFGEdgeRow) -> tuple[object, ...]:
    """Serialize a DFGEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The DFG edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with dfg_edges INSERT order.
    """
    return (
        row["function_goid_h128"],
        row["src_block_id"],
        row["dst_block_id"],
        row["src_var"],
        row["dst_var"],
        row["edge_kind"],
        row["via_phi"],
        row["use_kind"],
    )


__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "DFGEdgeRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "SymbolUseRow",
    "call_graph_edge_to_tuple",
    "call_graph_node_to_tuple",
    "cfg_block_to_tuple",
    "cfg_edge_to_tuple",
    "dfg_edge_to_tuple",
    "import_edge_to_tuple",
    "import_module_to_tuple",
    "symbol_use_to_tuple",
]
