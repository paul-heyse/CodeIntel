"""CPG edge assembly exports."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import tabular_to_table
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_EDGES_TABLE_KEY
from codeintel.build.hamilton.native.graphs.cpg2.types import (
    _CpgCallWiringInputs,
    _CpgEdgeCoreInputs,
    _CpgFlowInputs,
    _CpgLinkInputs,
    _CpgOverlayBytecodeInputs,
    _CpgOverlayEdgeInputs,
    _CpgOverlayInspectCoreInputs,
    _CpgOverlayInspectInputs,
    _CpgOverlayInspectRuntimeInputs,
    _CpgOverlayScopeInputs,
    _CpgOverlaySymbolInputs,
    _CpgOverlaySyntaxCallInputs,
    _CpgSymbolInputs,
    _CpgSymbolOccurrenceInputs,
    _CpgSymbolRelationInputs,
    _CpgSymbolTableInputs,
    _CpgSyntaxNodeInputs,
)
from codeintel.build.tabular.types import InferableTabularInput


def cpg_edge_symbol_inputs(
    q__core__syntax_edges: InferableTabularInput,
    cpg_edge_symbol_occurrence_inputs: _CpgSymbolOccurrenceInputs,
    cpg_edge_symbol_relation_inputs: _CpgSymbolRelationInputs,
    cpg_edge_symbol_table_inputs: _CpgSymbolTableInputs,
) -> _CpgSymbolInputs:
    """Collect symbol-layer inputs for CPG edge assembly.

    Returns
    -------
    _CpgSymbolInputs
        Symbol inputs for CPG edge assembly.
    """
    return _CpgSymbolInputs(
        syntax_edges=tabular_to_table(q__core__syntax_edges),
        occ_syntax=cpg_edge_symbol_occurrence_inputs.occ_syntax,
        occ_span=cpg_edge_symbol_occurrence_inputs.occ_span,
        symbol_rels=cpg_edge_symbol_relation_inputs.symbol_rels,
        symbol_goid=cpg_edge_symbol_relation_inputs.symbol_goid,
        scip_symbols=cpg_edge_symbol_table_inputs.scip_symbols,
        scip_external_symbols=cpg_edge_symbol_table_inputs.scip_external_symbols,
    )


def cpg_edge_symbol_occurrence_inputs(
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
) -> _CpgSymbolOccurrenceInputs:
    """Collect SCIP occurrence inputs for CPG symbol edges.

    Returns
    -------
    _CpgSymbolOccurrenceInputs
        Occurrence inputs for CPG symbol edges.
    """
    return _CpgSymbolOccurrenceInputs(
        occ_syntax=tabular_to_table(q__core__scip_occurrence_syntax_xref),
        occ_span=tabular_to_table(q__core__scip_occurrence_span_xref),
    )


def cpg_edge_symbol_relation_inputs(
    q__core__scip_symbol_relationships: InferableTabularInput,
    q__core__scip_symbol_goid_xref: InferableTabularInput,
) -> _CpgSymbolRelationInputs:
    """Collect SCIP relationship inputs for CPG symbol edges.

    Returns
    -------
    _CpgSymbolRelationInputs
        Relationship inputs for CPG symbol edges.
    """
    return _CpgSymbolRelationInputs(
        symbol_rels=tabular_to_table(q__core__scip_symbol_relationships),
        symbol_goid=tabular_to_table(q__core__scip_symbol_goid_xref),
    )


def cpg_edge_symbol_table_inputs(
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__scip_external_symbols: InferableTabularInput,
) -> _CpgSymbolTableInputs:
    """Collect SCIP symbol tables for CPG symbol edges.

    Returns
    -------
    _CpgSymbolTableInputs
        Symbol table inputs for CPG symbol edges.
    """
    return _CpgSymbolTableInputs(
        scip_symbols=tabular_to_table(q__core__scip_symbol_information),
        scip_external_symbols=tabular_to_table(q__core__scip_external_symbols),
    )


def cpg_edge_flow_inputs(
    q__core__goids: InferableTabularInput,
    q__graph__cfg_edges: InferableTabularInput,
    q__graph__dfg_edges: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__cdg_edges: InferableTabularInput,
) -> _CpgFlowInputs:
    """Collect flow-layer inputs for CPG edge assembly.

    Returns
    -------
    _CpgFlowInputs
        Flow inputs for CPG edge assembly.
    """
    return _CpgFlowInputs(
        goids=tabular_to_table(q__core__goids),
        cfg_edges=tabular_to_table(q__graph__cfg_edges),
        dfg_edges=tabular_to_table(q__graph__dfg_edges),
        cfg_blocks=tabular_to_table(q__graph__cfg_blocks),
        cdg_edges=tabular_to_table(q__graph__cdg_edges),
    )


def cpg_edge_link_inputs(
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__import_graph_edges: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
) -> _CpgLinkInputs:
    """Collect graph-link inputs for CPG edge assembly.

    Returns
    -------
    _CpgLinkInputs
        Graph-link inputs for CPG edge assembly.
    """
    return _CpgLinkInputs(
        call_edges=tabular_to_table(q__graph__call_graph_edges),
        import_edges=tabular_to_table(q__graph__import_graph_edges),
        import_modules=tabular_to_table(q__graph__import_modules),
    )


def cpg_edge_call_wiring_inputs(
    q__graph__cpg_edges_calls: InferableTabularInput,
    q__graph__cpg_edges_arg_to_param: InferableTabularInput,
    q__graph__cpg_edges_ret_to_call: InferableTabularInput,
) -> _CpgCallWiringInputs:
    """Collect call wiring inputs for CPG edge assembly.

    Returns
    -------
    _CpgCallWiringInputs
        Call wiring inputs for CPG edge assembly.
    """
    return _CpgCallWiringInputs(
        call_edges=tabular_to_table(q__graph__cpg_edges_calls),
        arg_to_param_edges=tabular_to_table(q__graph__cpg_edges_arg_to_param),
        ret_to_call_edges=tabular_to_table(q__graph__cpg_edges_ret_to_call),
    )


def cpg_edge_syntax_node_inputs(
    q__core__syntax_nodes: InferableTabularInput,
) -> _CpgSyntaxNodeInputs:
    """Collect syntax node inputs for CPG edge assembly.

    Returns
    -------
    _CpgSyntaxNodeInputs
        Syntax node inputs for CPG edge assembly.
    """
    return _CpgSyntaxNodeInputs(
        syntax_nodes=tabular_to_table(q__core__syntax_nodes),
    )


def cpg_edge_overlay_scope_inputs(
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_sym_bindings: InferableTabularInput,
    q__core__py_sym_scope_edges: InferableTabularInput,
    q__core__py_sym_namespace_edges: InferableTabularInput,
    q__core__py_sym_resolution_edges: InferableTabularInput,
) -> _CpgOverlayScopeInputs:
    """Collect scope overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayScopeInputs
        Scope overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayScopeInputs(
        py_sym_scopes=tabular_to_table(q__core__py_sym_scopes),
        py_sym_bindings=tabular_to_table(q__core__py_sym_bindings),
        py_sym_scope_edges=tabular_to_table(q__core__py_sym_scope_edges),
        py_sym_namespace_edges=tabular_to_table(q__core__py_sym_namespace_edges),
        py_sym_resolution_edges=tabular_to_table(q__core__py_sym_resolution_edges),
    )


def cpg_edge_overlay_symbol_inputs(
    q__core__ast_nodes: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    cpg_edge_overlay_scope_inputs: _CpgOverlayScopeInputs,
) -> _CpgOverlaySymbolInputs:
    """Collect symbol overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlaySymbolInputs
        Symbol overlay inputs for CPG edge assembly.
    """
    return _CpgOverlaySymbolInputs(
        ast_nodes=tabular_to_table(q__core__ast_nodes),
        scip_symbols=tabular_to_table(q__core__scip_symbol_information),
        scope_inputs=cpg_edge_overlay_scope_inputs,
    )


def cpg_edge_overlay_bytecode_inputs(
    q__core__py_bc_code_units: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__py_bc_cfg_edges: InferableTabularInput,
    q__core__py_bc_defuse_events: InferableTabularInput,
) -> _CpgOverlayBytecodeInputs:
    """Collect bytecode overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayBytecodeInputs
        Bytecode overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayBytecodeInputs(
        py_bc_code_units=tabular_to_table(q__core__py_bc_code_units),
        py_bc_instructions=tabular_to_table(q__core__py_bc_instructions),
        py_bc_blocks=tabular_to_table(q__core__py_bc_blocks),
        py_bc_cfg_edges=tabular_to_table(q__core__py_bc_cfg_edges),
        py_bc_defuse_events=tabular_to_table(q__core__py_bc_defuse_events),
    )


def cpg_edge_overlay_syntax_call_inputs(
    q__core__syntax_calls: InferableTabularInput,
    q__core__syntax_call_args: InferableTabularInput,
) -> _CpgOverlaySyntaxCallInputs:
    """Collect syntax call inputs for CPG inspect overlays.

    Returns
    -------
    _CpgOverlaySyntaxCallInputs
        Syntax call inputs for inspect overlays.
    """
    return _CpgOverlaySyntaxCallInputs(
        syntax_calls=tabular_to_table(q__core__syntax_calls),
        syntax_call_args=tabular_to_table(q__core__syntax_call_args),
    )


def cpg_edge_overlay_inspect_core_inputs(
    q__core__py_inspect_objects: InferableTabularInput,
    q__core__py_inspect_class_mro: InferableTabularInput,
    q__core__py_inspect_class_attrs: InferableTabularInput,
    q__core__py_inspect_unwrap_hops: InferableTabularInput,
    q__core__py_inspect_source: InferableTabularInput,
) -> _CpgOverlayInspectCoreInputs:
    """Collect core inspect inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectCoreInputs
        Core inspect inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectCoreInputs(
        py_inspect_objects=tabular_to_table(q__core__py_inspect_objects),
        py_inspect_class_mro=tabular_to_table(q__core__py_inspect_class_mro),
        py_inspect_class_attrs=tabular_to_table(q__core__py_inspect_class_attrs),
        py_inspect_unwrap_hops=tabular_to_table(q__core__py_inspect_unwrap_hops),
        py_inspect_source=tabular_to_table(q__core__py_inspect_source),
    )


def cpg_edge_overlay_inspect_runtime_inputs(
    q__core__py_inspect_signatures: InferableTabularInput,
    q__core__py_inspect_signature_params: InferableTabularInput,
    q__core__py_inspect_runtime_state: InferableTabularInput,
) -> _CpgOverlayInspectRuntimeInputs:
    """Collect runtime inspect inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectRuntimeInputs
        Runtime inspect inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectRuntimeInputs(
        py_inspect_signatures=tabular_to_table(q__core__py_inspect_signatures),
        py_inspect_signature_params=tabular_to_table(q__core__py_inspect_signature_params),
        py_inspect_runtime_state=tabular_to_table(q__core__py_inspect_runtime_state),
    )


def cpg_edge_overlay_inspect_inputs(
    cpg_edge_overlay_inspect_core_inputs: _CpgOverlayInspectCoreInputs,
    cpg_edge_overlay_inspect_runtime_inputs: _CpgOverlayInspectRuntimeInputs,
    cpg_edge_overlay_syntax_call_inputs: _CpgOverlaySyntaxCallInputs,
) -> _CpgOverlayInspectInputs:
    """Collect inspect overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectInputs
        Inspect overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectInputs(
        py_inspect_objects=cpg_edge_overlay_inspect_core_inputs.py_inspect_objects,
        py_inspect_class_mro=cpg_edge_overlay_inspect_core_inputs.py_inspect_class_mro,
        py_inspect_class_attrs=cpg_edge_overlay_inspect_core_inputs.py_inspect_class_attrs,
        py_inspect_unwrap_hops=cpg_edge_overlay_inspect_core_inputs.py_inspect_unwrap_hops,
        py_inspect_signatures=cpg_edge_overlay_inspect_runtime_inputs.py_inspect_signatures,
        py_inspect_signature_params=(
            cpg_edge_overlay_inspect_runtime_inputs.py_inspect_signature_params
        ),
        py_inspect_source=cpg_edge_overlay_inspect_core_inputs.py_inspect_source,
        py_inspect_runtime_state=cpg_edge_overlay_inspect_runtime_inputs.py_inspect_runtime_state,
        syntax_calls=cpg_edge_overlay_syntax_call_inputs.syntax_calls,
        syntax_call_args=cpg_edge_overlay_syntax_call_inputs.syntax_call_args,
    )


def cpg_edge_overlay_inputs(
    cpg_edge_overlay_symbol_inputs: _CpgOverlaySymbolInputs,
    cpg_edge_overlay_bytecode_inputs: _CpgOverlayBytecodeInputs,
    cpg_edge_overlay_inspect_inputs: _CpgOverlayInspectInputs,
) -> _CpgOverlayEdgeInputs:
    """Collect overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayEdgeInputs
        Overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayEdgeInputs(
        ast_nodes=cpg_edge_overlay_symbol_inputs.ast_nodes,
        syntax_calls=cpg_edge_overlay_inspect_inputs.syntax_calls,
        syntax_call_args=cpg_edge_overlay_inspect_inputs.syntax_call_args,
        scip_symbols=cpg_edge_overlay_symbol_inputs.scip_symbols,
        py_sym_scopes=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_scopes,
        py_sym_bindings=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_bindings,
        py_sym_scope_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_scope_edges,
        py_sym_namespace_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_namespace_edges,
        py_sym_resolution_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_resolution_edges,
        py_bc_code_units=cpg_edge_overlay_bytecode_inputs.py_bc_code_units,
        py_bc_instructions=cpg_edge_overlay_bytecode_inputs.py_bc_instructions,
        py_bc_blocks=cpg_edge_overlay_bytecode_inputs.py_bc_blocks,
        py_bc_cfg_edges=cpg_edge_overlay_bytecode_inputs.py_bc_cfg_edges,
        py_bc_defuse_events=cpg_edge_overlay_bytecode_inputs.py_bc_defuse_events,
        py_inspect_objects=cpg_edge_overlay_inspect_inputs.py_inspect_objects,
        py_inspect_class_mro=cpg_edge_overlay_inspect_inputs.py_inspect_class_mro,
        py_inspect_class_attrs=cpg_edge_overlay_inspect_inputs.py_inspect_class_attrs,
        py_inspect_unwrap_hops=cpg_edge_overlay_inspect_inputs.py_inspect_unwrap_hops,
        py_inspect_signatures=cpg_edge_overlay_inspect_inputs.py_inspect_signatures,
        py_inspect_signature_params=cpg_edge_overlay_inspect_inputs.py_inspect_signature_params,
        py_inspect_source=cpg_edge_overlay_inspect_inputs.py_inspect_source,
        py_inspect_runtime_state=cpg_edge_overlay_inspect_inputs.py_inspect_runtime_state,
    )


def cpg_edge_core_inputs(
    cpg_edge_symbol_inputs: _CpgSymbolInputs,
    cpg_edge_flow_inputs: _CpgFlowInputs,
    cpg_edge_link_inputs: _CpgLinkInputs,
    cpg_edge_call_wiring_inputs: _CpgCallWiringInputs,
    cpg_edge_syntax_node_inputs: _CpgSyntaxNodeInputs,
) -> _CpgEdgeCoreInputs:
    """Collect core edge inputs for CPG edge assembly.

    Returns
    -------
    _CpgEdgeCoreInputs
        Core inputs for CPG edge assembly.
    """
    return _CpgEdgeCoreInputs(
        symbol=cpg_edge_symbol_inputs,
        flow=cpg_edge_flow_inputs,
        link=cpg_edge_link_inputs,
        call_wiring=cpg_edge_call_wiring_inputs,
        syntax_nodes=cpg_edge_syntax_node_inputs,
    )


def cpg_edges(
    cpg2_edges__frames: pa.Table,
) -> InferableTabularInput:
    """Build CPG edges from syntax, symbol, and flow sources.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.cpg_edges.
    """
    return cpg2_edges__frames


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
    "cpg_edge_symbol_occurrence_inputs",
    "cpg_edge_symbol_relation_inputs",
    "cpg_edge_symbol_table_inputs",
    "cpg_edge_syntax_node_inputs",
    "cpg_edges",
]
