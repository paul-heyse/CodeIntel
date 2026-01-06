"""Shared dataclasses for CPG assembly inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.hamilton.native.options.graphs import CpgOptions
from codeintel.build.tabular.types import InferableTabularInput


@dataclass(frozen=True, slots=True)
class CpgOverlayOptions:
    """Enablement flags for optional CPG overlays."""

    enable_symtable: bool
    enable_bytecode: bool
    enable_inspect: bool
    inspect_allowlist: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CpgEdgeConfig:
    """Configuration bundle for CPG edge assembly."""

    overlay_options: CpgOverlayOptions
    options: CpgOptions


@dataclass(frozen=True)
class _CpgSymbolInputs:
    syntax_edges: pa.Table
    occ_syntax: pa.Table
    occ_span: pa.Table
    symbol_rels: pa.Table
    symbol_goid: pa.Table


@dataclass(frozen=True)
class _CpgFlowInputs:
    goids: pa.Table
    cfg_edges: pa.Table
    dfg_edges: pa.Table
    cfg_blocks: pa.Table
    cdg_edges: pa.Table


@dataclass(frozen=True)
class _CpgNodeCoreInputs:
    syntax_nodes: InferableTabularInput
    ast_nodes: InferableTabularInput
    scip_symbol_information: InferableTabularInput
    goids: InferableTabularInput
    py_sym_scopes: InferableTabularInput
    py_sym_bindings: InferableTabularInput
    py_bc_code_units: InferableTabularInput
    py_bc_instructions: InferableTabularInput
    py_bc_blocks: InferableTabularInput
    py_inspect_objects: InferableTabularInput
    py_inspect_signatures: InferableTabularInput
    py_inspect_signature_params: InferableTabularInput
    ts_tokens: InferableTabularInput
    ts_trivia: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeSyntaxInputs:
    syntax_nodes: InferableTabularInput
    ast_nodes: InferableTabularInput
    scip_symbol_information: InferableTabularInput
    goids: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodePyInputs:
    py_sym_scopes: InferableTabularInput
    py_sym_bindings: InferableTabularInput
    py_bc_code_units: InferableTabularInput
    py_bc_instructions: InferableTabularInput
    py_bc_blocks: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeInspectInputs:
    py_inspect_objects: InferableTabularInput
    py_inspect_signatures: InferableTabularInput
    py_inspect_signature_params: InferableTabularInput
    ts_tokens: InferableTabularInput
    ts_trivia: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeGraphInputs:
    cfg_blocks: InferableTabularInput
    import_modules: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeInputs:
    core: _CpgNodeCoreInputs
    graph: _CpgNodeGraphInputs


@dataclass(frozen=True)
class _CpgNodeCoreLazyFrames:
    syntax_nodes: pa.Table
    ast_nodes: pa.Table
    scip_symbol_information: pa.Table
    goids: pa.Table
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_inspect_objects: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    ts_tokens: pa.Table
    ts_trivia: pa.Table


@dataclass(frozen=True)
class _CpgNodeGraphLazyFrames:
    cfg_blocks: pa.Table
    import_modules: pa.Table


@dataclass(frozen=True)
class _CpgLinkInputs:
    call_edges: pa.Table
    import_edges: pa.Table
    import_modules: pa.Table


@dataclass(frozen=True)
class _CpgCallWiringInputs:
    call_edges: pa.Table
    arg_to_param_edges: pa.Table
    ret_to_call_edges: pa.Table


@dataclass(frozen=True)
class _CpgSyntaxNodeInputs:
    syntax_nodes: pa.Table


@dataclass(frozen=True)
class _CpgOverlayEdgeInputs:
    ast_nodes: pa.Table
    syntax_calls: pa.Table
    syntax_call_args: pa.Table
    scip_symbols: pa.Table
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_sym_scope_edges: pa.Table
    py_sym_namespace_edges: pa.Table
    py_sym_resolution_edges: pa.Table
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_bc_cfg_edges: pa.Table
    py_bc_defuse_events: pa.Table
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_source: pa.Table
    py_inspect_runtime_state: pa.Table


@dataclass(frozen=True)
class _CpgOverlayScopeInputs:
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_sym_scope_edges: pa.Table
    py_sym_namespace_edges: pa.Table
    py_sym_resolution_edges: pa.Table


@dataclass(frozen=True)
class _CpgOverlaySymbolInputs:
    ast_nodes: pa.Table
    scip_symbols: pa.Table
    scope_inputs: _CpgOverlayScopeInputs


@dataclass(frozen=True)
class _CpgOverlayBytecodeInputs:
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_bc_cfg_edges: pa.Table
    py_bc_defuse_events: pa.Table


@dataclass(frozen=True)
class _CpgOverlaySyntaxCallInputs:
    syntax_calls: pa.Table
    syntax_call_args: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectCoreInputs:
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_source: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectRuntimeInputs:
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_runtime_state: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectInputs:
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_source: pa.Table
    py_inspect_runtime_state: pa.Table
    syntax_calls: pa.Table
    syntax_call_args: pa.Table


@dataclass(frozen=True)
class _CpgEdgeCoreInputs:
    symbol: _CpgSymbolInputs
    flow: _CpgFlowInputs
    link: _CpgLinkInputs
    call_wiring: _CpgCallWiringInputs
    syntax_nodes: _CpgSyntaxNodeInputs


__all__ = [
    "CpgEdgeConfig",
    "CpgOverlayOptions",
    "_CpgCallWiringInputs",
    "_CpgEdgeCoreInputs",
    "_CpgFlowInputs",
    "_CpgLinkInputs",
    "_CpgNodeCoreInputs",
    "_CpgNodeCoreLazyFrames",
    "_CpgNodeGraphInputs",
    "_CpgNodeGraphLazyFrames",
    "_CpgNodeInputs",
    "_CpgNodeInspectInputs",
    "_CpgNodePyInputs",
    "_CpgNodeSyntaxInputs",
    "_CpgOverlayBytecodeInputs",
    "_CpgOverlayEdgeInputs",
    "_CpgOverlayInspectCoreInputs",
    "_CpgOverlayInspectInputs",
    "_CpgOverlayInspectRuntimeInputs",
    "_CpgOverlayScopeInputs",
    "_CpgOverlaySymbolInputs",
    "_CpgOverlaySyntaxCallInputs",
    "_CpgSymbolInputs",
    "_CpgSyntaxNodeInputs",
]
