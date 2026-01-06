"""Bytecode overlay exports for CPG."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.cpg.constants import PY_BC_INSTRUCTIONS_TABLE_KEY
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id
from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_bytecode import (
    cpg2_edges__py_bc_callsite,
    cpg2_edges__py_bc_callsite_symbol,
    cpg2_edges__py_bc_stack,
)


def instruction_cpg_id(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    code_unit_id: str,
    instr_id: str,
) -> int:
    """Public wrapper for instruction CPG node IDs.

    Returns
    -------
    int
        Stable CPG node identifier.
    """
    pk: Mapping[str, object] = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "code_unit_id": code_unit_id,
        "instr_id": instr_id,
    }
    return cpg_node_id(PY_BC_INSTRUCTIONS_TABLE_KEY, pk)


def py_bc_callsite_symbol_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode callsite symbol edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of callsite symbol edges.
    """
    return cpg2_edges__py_bc_callsite_symbol(instructions, syntax_calls, scip_symbols)


def py_bc_callsite_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode callsite edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of callsite edges.
    """
    return cpg2_edges__py_bc_callsite(instructions, syntax_calls)


def py_bc_stack_edges_to_cpg(
    instructions: pa.Table,
    blocks: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode stack edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of stack edges.
    """
    return cpg2_edges__py_bc_stack(instructions, blocks)


__all__ = [
    "instruction_cpg_id",
    "py_bc_callsite_edges_to_cpg",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
]
