"""CPG node assembly exports."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_NODES_TABLE_KEY
from codeintel.build.hamilton.native.graphs.cpg2.types import (
    _CpgNodeCoreInputs,
    _CpgNodeGraphInputs,
    _CpgNodeInputs,
    _CpgNodeInspectInputs,
    _CpgNodePyInputs,
    _CpgNodeSyntaxInputs,
)
from codeintel.build.tabular.types import InferableTabularInput


def cpg_nodes__syntax_inputs(
    q__core__syntax_nodes: InferableTabularInput,
    q__core__ast_nodes: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> _CpgNodeSyntaxInputs:
    """Collect syntax-layer inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeSyntaxInputs
        Syntax inputs for CPG node assembly.
    """
    return _CpgNodeSyntaxInputs(
        syntax_nodes=q__core__syntax_nodes,
        ast_nodes=q__core__ast_nodes,
        scip_symbol_information=q__core__scip_symbol_information,
        goids=q__core__goids,
    )


def cpg_nodes__py_inputs(
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_sym_bindings: InferableTabularInput,
    q__core__py_bc_code_units: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
) -> _CpgNodePyInputs:
    """Collect Python overlay inputs for CPG node assembly.

    Returns
    -------
    _CpgNodePyInputs
        Python overlay inputs for CPG node assembly.
    """
    return _CpgNodePyInputs(
        py_sym_scopes=q__core__py_sym_scopes,
        py_sym_bindings=q__core__py_sym_bindings,
        py_bc_code_units=q__core__py_bc_code_units,
        py_bc_instructions=q__core__py_bc_instructions,
        py_bc_blocks=q__core__py_bc_blocks,
    )


def cpg_nodes__inspect_inputs(
    q__core__py_inspect_objects: InferableTabularInput,
    q__core__py_inspect_signatures: InferableTabularInput,
    q__core__py_inspect_signature_params: InferableTabularInput,
    q__core__ts_tokens: InferableTabularInput,
    q__core__ts_trivia: InferableTabularInput,
) -> _CpgNodeInspectInputs:
    """Collect inspect inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeInspectInputs
        Inspect inputs for CPG node assembly.
    """
    return _CpgNodeInspectInputs(
        py_inspect_objects=q__core__py_inspect_objects,
        py_inspect_signatures=q__core__py_inspect_signatures,
        py_inspect_signature_params=q__core__py_inspect_signature_params,
        ts_tokens=q__core__ts_tokens,
        ts_trivia=q__core__ts_trivia,
    )


def cpg_nodes__core_inputs(
    cpg_nodes__syntax_inputs: _CpgNodeSyntaxInputs,
    cpg_nodes__py_inputs: _CpgNodePyInputs,
    cpg_nodes__inspect_inputs: _CpgNodeInspectInputs,
) -> _CpgNodeCoreInputs:
    """Collect core inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeCoreInputs
        Core inputs for CPG node assembly.
    """
    return _CpgNodeCoreInputs(
        syntax_nodes=cpg_nodes__syntax_inputs.syntax_nodes,
        ast_nodes=cpg_nodes__syntax_inputs.ast_nodes,
        scip_symbol_information=cpg_nodes__syntax_inputs.scip_symbol_information,
        goids=cpg_nodes__syntax_inputs.goids,
        py_sym_scopes=cpg_nodes__py_inputs.py_sym_scopes,
        py_sym_bindings=cpg_nodes__py_inputs.py_sym_bindings,
        py_bc_code_units=cpg_nodes__py_inputs.py_bc_code_units,
        py_bc_instructions=cpg_nodes__py_inputs.py_bc_instructions,
        py_bc_blocks=cpg_nodes__py_inputs.py_bc_blocks,
        py_inspect_objects=cpg_nodes__inspect_inputs.py_inspect_objects,
        py_inspect_signatures=cpg_nodes__inspect_inputs.py_inspect_signatures,
        py_inspect_signature_params=cpg_nodes__inspect_inputs.py_inspect_signature_params,
        ts_tokens=cpg_nodes__inspect_inputs.ts_tokens,
        ts_trivia=cpg_nodes__inspect_inputs.ts_trivia,
    )


def cpg_nodes__graph_inputs(
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
) -> _CpgNodeGraphInputs:
    """Collect graph inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeGraphInputs
        Graph inputs for CPG node assembly.
    """
    return _CpgNodeGraphInputs(
        cfg_blocks=q__graph__cfg_blocks,
        import_modules=q__graph__import_modules,
    )


def cpg_nodes__inputs(
    cpg_nodes__core_inputs: _CpgNodeCoreInputs,
    cpg_nodes__graph_inputs: _CpgNodeGraphInputs,
) -> _CpgNodeInputs:
    """Collect all inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeInputs
        Inputs for CPG node assembly.
    """
    return _CpgNodeInputs(
        core=cpg_nodes__core_inputs,
        graph=cpg_nodes__graph_inputs,
    )


def cpg_nodes(
    cpg2_nodes__frames: pa.Table,
) -> InferableTabularInput:
    """Build CPG nodes from syntax, symbol, and flow inventories.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.cpg_nodes.
    """
    return cpg2_nodes__frames


__all__ = [
    "CPG_NODES_TABLE_KEY",
    "cpg_nodes",
    "cpg_nodes__core_inputs",
    "cpg_nodes__graph_inputs",
    "cpg_nodes__inputs",
    "cpg_nodes__inspect_inputs",
    "cpg_nodes__py_inputs",
    "cpg_nodes__syntax_inputs",
]
