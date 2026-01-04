"""Bytecode overlay exports for CPG."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.cpg import _legacy

instruction_cpg_id = _legacy.instruction_cpg_id
py_bc_callsite_edges_to_cpg = _legacy.py_bc_callsite_edges_to_cpg
py_bc_callsite_symbol_edges_to_cpg = _legacy.py_bc_callsite_symbol_edges_to_cpg
py_bc_stack_edges_to_cpg = _legacy.py_bc_stack_edges_to_cpg

__all__ = [
    "instruction_cpg_id",
    "py_bc_callsite_edges_to_cpg",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
]
