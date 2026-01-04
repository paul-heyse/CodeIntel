"""Tests for bytecode-specific CPG edges."""

from __future__ import annotations

from typing import cast

import pyarrow as pa
import pytest

from codeintel.build.hamilton.native.graphs.cpg import (
    SCIP_SYMBOLS_TABLE_KEY,
    instruction_cpg_id,
    py_bc_callsite_symbol_edges_to_cpg,
    py_bc_stack_edges_to_cpg,
    stable_cpg_id,
)
from codeintel.core.serialization.payload import decode_payload

pytestmark = pytest.mark.no_runtime_env

EXPECTED_STACK_EDGE_COUNT = 2
EXPECTED_CALLSITE_EDGE_COUNT = 1


def test_py_bc_stack_edges_basic() -> None:
    """Build basic stack edges for a minimal bytecode sequence."""
    instructions = pa.Table.from_pylist(
        [
            {
                "repo": "repo",
                "commit": "commit",
                "rel_path": "src/app.py",
                "code_unit_id": "unit1",
                "instr_id": "i0",
                "instr_index": 0,
                "opname": "LOAD_CONST",
                "baseopname": None,
                "arg": 0,
            },
            {
                "repo": "repo",
                "commit": "commit",
                "rel_path": "src/app.py",
                "code_unit_id": "unit1",
                "instr_id": "i1",
                "instr_index": 1,
                "opname": "LOAD_CONST",
                "baseopname": None,
                "arg": 0,
            },
            {
                "repo": "repo",
                "commit": "commit",
                "rel_path": "src/app.py",
                "code_unit_id": "unit1",
                "instr_id": "i2",
                "instr_index": 2,
                "opname": "BINARY_OP",
                "baseopname": None,
                "arg": 0,
            },
        ]
    )
    blocks = pa.Table.from_pylist(
        [
            {
                "code_unit_id": "unit1",
                "block_id": "b0",
                "first_instr_index": 0,
                "last_instr_index": 2,
            }
        ]
    )

    edges = py_bc_stack_edges_to_cpg(instructions, blocks)
    assert edges.num_rows == EXPECTED_STACK_EDGE_COUNT
    assert set(edges.column("edge_kind").to_pylist()) == {"STACK_REACHES"}
    expected_src_ids = {
        instruction_cpg_id(
            repo="repo",
            commit="commit",
            rel_path="src/app.py",
            code_unit_id="unit1",
            instr_id="i0",
        ),
        instruction_cpg_id(
            repo="repo",
            commit="commit",
            rel_path="src/app.py",
            code_unit_id="unit1",
            instr_id="i1",
        ),
    }
    assert set(edges.column("src_cpg_node_id").to_pylist()) == expected_src_ids
    payloads = [
        cast("dict[str, object]", decode_payload(row["extras_json"])) for row in edges.to_pylist()
    ]
    assert {payload["stack_pop_index"] for payload in payloads} == {0, 1}
    assert {payload["src_opname"] for payload in payloads} == {"LOAD_CONST"}
    assert {payload["dst_opname"] for payload in payloads} == {"BINARY_OP"}


def test_py_bc_callsite_symbol_edges_match_display_name() -> None:
    """Create callsite edges that resolve by display name."""
    instructions = pa.Table.from_pylist(
        [
            {
                "repo": "repo",
                "commit": "commit",
                "rel_path": "src/app.py",
                "code_unit_id": "unit1",
                "instr_id": "call0",
                "baseopname": "CALL",
                "opname": "CALL",
                "span_start_byte": 12,
                "span_end_byte": 18,
                "line_number": 1,
            }
        ]
    )
    syntax_calls = pa.Table.from_pylist(
        [
            {
                "repo": "repo",
                "commit": "commit",
                "rel_path": "src/app.py",
                "producer": "ast",
                "call_id": "call-1",
                "call_node_id": "node-1",
                "start_byte": 10,
                "end_byte": 20,
                "start_line": 1,
                "end_line": 1,
                "callee_text": "module.func",
            }
        ]
    )
    scip_symbols = pa.Table.from_pylist(
        [
            {
                "repo": "repo",
                "commit": "commit",
                "symbol": "sym1",
                "display_name": "module.func",
            }
        ]
    )

    edges = py_bc_callsite_symbol_edges_to_cpg(instructions, syntax_calls, scip_symbols)
    assert edges.num_rows == EXPECTED_CALLSITE_EDGE_COUNT
    row = edges.to_pylist()[0]
    assert row["edge_kind"] == "BYTECODE_CALLS_SYMBOL"
    assert row["edge_layer"] == "CALL"
    assert row["src_cpg_node_id"] == instruction_cpg_id(
        repo="repo",
        commit="commit",
        rel_path="src/app.py",
        code_unit_id="unit1",
        instr_id="call0",
    )
    expected_dst = stable_cpg_id(
        SCIP_SYMBOLS_TABLE_KEY,
        {"repo": "repo", "commit": "commit", "symbol": "sym1"},
    )
    assert row["dst_cpg_node_id"] == expected_dst
    payload = cast("dict[str, object]", decode_payload(row["extras_json"]))
    assert payload["match_kind"] == "display_name"
    assert payload["callee_text"] == "module.func"
