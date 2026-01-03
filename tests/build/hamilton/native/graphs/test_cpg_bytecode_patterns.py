"""Tests for bytecode stack and callsite edges across control-flow patterns."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from codeintel.build.hamilton.native.graphs.cpg import (
    py_bc_callsite_edges_to_cpg,
    py_bc_stack_edges_to_cpg,
)
from codeintel.core.serialization.payload import decode_payload
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.compute.dis_extract import DisExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree

pytestmark = pytest.mark.no_runtime_env


def _reader_to_table(reader: pa.RecordBatchReader) -> pa.Table:
    return reader.read_all()


def _callee_leaf(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.rsplit(".", 1)[-1]


def test_cpg_bytecode_edges_patterns(tmp_path: Path) -> None:
    """Ensure stack/callsite edges capture common control-flow patterns."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/patterns.py": "\n".join(
                [
                    "def call_if(x):",
                    "    return x",
                    "",
                    "def call_loop(x):",
                    "    return x",
                    "",
                    "def call_with(x):",
                    "    return x",
                    "",
                    "def call_except():",
                    "    return None",
                    "",
                    "def call_finally():",
                    "    return None",
                    "",
                    "def call_comp(x):",
                    "    return x",
                    "",
                    "def call_decorated(x):",
                    "    return x",
                    "",
                    "def deco(fn):",
                    "    def wrapper(*args, **kwargs):",
                    "        return fn(*args, **kwargs)",
                    "    return wrapper",
                    "",
                    "@deco",
                    "def decorated(x):",
                    "    return call_decorated(x)",
                    "",
                    "def pattern(flag):",
                    "    if flag:",
                    "        call_if(1)",
                    "    for i in range(2):",
                    "        call_loop(i)",
                    "    try:",
                    "        with open(\"x\") as handle:",
                    "            call_with(handle.read())",
                    "    except OSError:",
                    "        call_except()",
                    "    finally:",
                    "        call_finally()",
                    "    return [call_comp(i) for i in range(2)]",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    discovery = FilesystemDiscoveryAdapter(repo_root)

    dis_result = DisExtractStep(discovery).execute(modules, repo="demo", commit="abc123")
    cst_result = CstExtractStep(discovery).execute(modules, repo="demo", commit="abc123")
    assert dis_result.result.success
    assert cst_result.result.success

    instructions = _reader_to_table(dis_result.instruction_rows_reader)
    blocks = _reader_to_table(dis_result.block_rows_reader)
    syntax_calls = _reader_to_table(cst_result.syntax_calls_rows_reader)

    stack_edges = py_bc_stack_edges_to_cpg(instructions, blocks)
    assert stack_edges.num_rows > 0

    callsite_edges = py_bc_callsite_edges_to_cpg(instructions, syntax_calls)
    assert callsite_edges.num_rows > 0

    payloads = [decode_payload(row.get("extras_json")) for row in callsite_edges.to_pylist()]
    callees = {
        _callee_leaf(payload.get("callee_text"))
        for payload in payloads
        if isinstance(payload, dict)
    }
    expected = {
        "call_if",
        "call_loop",
        "call_with",
        "call_except",
        "call_finally",
        "call_comp",
        "deco",
        "call_decorated",
    }
    assert expected.issubset(callees)
