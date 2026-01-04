"""CPG node assembly exports."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.cpg import _legacy

CPG_NODES_TABLE_KEY = _legacy.CPG_NODES_TABLE_KEY

cpg_nodes__syntax_inputs = _legacy.cpg_nodes__syntax_inputs
cpg_nodes__py_inputs = _legacy.cpg_nodes__py_inputs
cpg_nodes__inspect_inputs = _legacy.cpg_nodes__inspect_inputs
cpg_nodes__core_inputs = _legacy.cpg_nodes__core_inputs
cpg_nodes__graph_inputs = _legacy.cpg_nodes__graph_inputs
cpg_nodes__inputs = _legacy.cpg_nodes__inputs
cpg_nodes = _legacy.cpg_nodes

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
