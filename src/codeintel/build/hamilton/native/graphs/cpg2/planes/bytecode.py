"""Python bytecode plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.serialization.payload import encode_payload

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"


def cpg2_nodes__py_bc_code_units(code_units: pa.Table) -> pa.Table:
    """Build CPG nodes for bytecode code units.

    Returns
    -------
    pyarrow.Table
        CPG node table for code units.
    """
    required = {"repo", "commit", "rel_path", "code_unit_id"}
    if not required.issubset(set(code_units.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(code_units):
        pk_values = pk_from_row(row, table_key=PY_BC_CODE_UNITS_TABLE_KEY)
        extras_values = {
            "qualpath": row.get("qualpath"),
            "co_name": row.get("co_name"),
            "co_qualname": row.get("co_qualname"),
            "kind": row.get("kind"),
            "co_firstlineno": row.get("co_firstlineno"),
            "flags": row.get("flags"),
            "argcount": row.get("argcount"),
            "posonlyargcount": row.get("posonlyargcount"),
            "kwonlyargcount": row.get("kwonlyargcount"),
            "nlocals": row.get("nlocals"),
            "stacksize": row.get("stacksize"),
            "varnames": row.get("varnames"),
            "names": row.get("names"),
            "freevars": row.get("freevars"),
            "cellvars": row.get("cellvars"),
            "bytecode_len": row.get("bytecode_len"),
            "exceptiontable_len": row.get("exceptiontable_len"),
            "python_version": row.get("python_version"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_BC_CODE_UNITS_TABLE_KEY, pk_values),
                "node_kind": "BC_CODE_UNIT",
                "source_table_key": PY_BC_CODE_UNITS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_bc_instructions(instructions: pa.Table) -> pa.Table:
    """Build CPG nodes for bytecode instructions.

    Returns
    -------
    pyarrow.Table
        CPG node table for instructions.
    """
    required = {"repo", "commit", "rel_path", "code_unit_id", "instr_id"}
    if not required.issubset(set(instructions.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(instructions):
        pk_values = pk_from_row(row, table_key=PY_BC_INSTRUCTIONS_TABLE_KEY)
        extras_values = {
            "instr_index": row.get("instr_index"),
            "start_offset": row.get("start_offset"),
            "offset": row.get("offset"),
            "end_offset": row.get("end_offset"),
            "opcode": row.get("opcode"),
            "opname": row.get("opname"),
            "baseopname": row.get("baseopname"),
            "arg": row.get("arg"),
            "argrepr": row.get("argrepr"),
            "argval_kind": row.get("argval_kind"),
            "argval_str": row.get("argval_str"),
            "argval_int": row.get("argval_int"),
            "argval_repr": row.get("argval_repr"),
            "is_jump_target": row.get("is_jump_target"),
            "jump_target_offset": row.get("jump_target_offset"),
            "jump_target_label": row.get("jump_target_label"),
            "label": row.get("label"),
            "starts_line": row.get("starts_line"),
            "line_number": row.get("line_number"),
            "pos": row.get("pos"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values),
                "node_kind": "BC_INSTR",
                "source_table_key": PY_BC_INSTRUCTIONS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_bc_blocks(blocks: pa.Table) -> pa.Table:
    """Build CPG nodes for bytecode blocks.

    Returns
    -------
    pyarrow.Table
        CPG node table for blocks.
    """
    required = {"repo", "commit", "rel_path", "block_id", "code_unit_id"}
    if not required.issubset(set(blocks.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(blocks):
        pk_values = pk_from_row(row, table_key=PY_BC_BLOCKS_TABLE_KEY)
        extras_values = {
            "code_unit_id": row.get("code_unit_id"),
            "start_offset": row.get("start_offset"),
            "end_offset": row.get("end_offset"),
            "start_label": row.get("start_label"),
            "kind": row.get("kind"),
            "first_instr_index": row.get("first_instr_index"),
            "last_instr_index": row.get("last_instr_index"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_BC_BLOCKS_TABLE_KEY, pk_values),
                "node_kind": "BC_BLOCK",
                "source_table_key": PY_BC_BLOCKS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("anchor_span_start_byte"),
                "end_byte": row.get("anchor_span_end_byte"),
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def _payload_bytes(values: dict[str, object]) -> bytes:
    encoded = encode_payload(values)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


__all__ = [
    "cpg2_nodes__py_bc_blocks",
    "cpg2_nodes__py_bc_code_units",
    "cpg2_nodes__py_bc_instructions",
]
