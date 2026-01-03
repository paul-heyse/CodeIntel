"""Tests for AST span merges into syntax facts."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.columnar.rows import ColumnarRows
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree


def _columnar_to_dicts(rows: ColumnarRows) -> list[dict[str, object]]:
    if not rows:
        return []
    columns = list(rows.keys())
    if not columns:
        return []
    row_count = len(rows[columns[0]])
    return [{col: rows[col][idx] for col in columns} for idx in range(row_count)]


def _has_ast_extras(rows: ColumnarRows) -> bool:
    for row in _columnar_to_dicts(rows):
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

    assert _has_ast_extras(result.syntax_defs_rows)
    assert _has_ast_extras(result.syntax_refs_rows)
    assert _has_ast_extras(result.syntax_imports_rows)
    assert _has_ast_extras(result.syntax_func_params_rows)
