"""Tests for AST span merges into syntax facts."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree


def _reader_to_dicts(reader: pa.RecordBatchReader) -> list[dict[str, object]]:
    table = pa.Table.from_batches(reader, schema=reader.schema)
    return list(table.to_pylist())


def _has_ast_extras(reader: pa.RecordBatchReader) -> bool:
    for row in _reader_to_dicts(reader):
        extras = row.get("extras_json")
        if isinstance(extras, dict) and "ast_node_id" in extras:
            return True
    return False


def test_ast_span_joins(tmp_path: Path) -> None:
    """Ensure AST extras are attached to syntax rows."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/sample.py": "\n".join(
                [
                    "import os",
                    "",
                    "def foo(a: int) -> int:",
                    "    b = a + 1",
                    "    return b",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = CstExtractStep(FilesystemDiscoveryAdapter(repo_root), emit_ast_nodes=True)
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    assert _has_ast_extras(result.syntax_defs_rows_reader)
    assert _has_ast_extras(result.syntax_refs_rows_reader)
    assert _has_ast_extras(result.syntax_imports_rows_reader)
    assert _has_ast_extras(result.syntax_func_params_rows_reader)
