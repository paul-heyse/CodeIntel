"""Tests for symtable extraction outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from codeintel.core.columnar.rows import ColumnarRows
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.symtable_extract import SymtableExtractStep
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


def test_symtable_resolution_edges(tmp_path: Path) -> None:
    """Ensure symtable extraction emits resolution edges."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/mod.py": "\n".join(
                [
                    "x = 1",
                    "",
                    "def outer() -> int:",
                    "    y = 2",
                    "    def inner() -> int:",
                    "        nonlocal y",
                    "        global x",
                    "        return x + y",
                    "    return inner()",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = SymtableExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    rows = _columnar_to_dicts(result.resolution_edge_rows)
    kinds = {row.get("kind") for row in rows if isinstance(row.get("kind"), str)}
    assert "GLOBAL" in kinds
    assert "NONLOCAL" in kinds


def test_symtable_freevars(tmp_path: Path) -> None:
    """Ensure freevars are recorded for nested scopes."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/mod.py": "\n".join(
                [
                    "def outer() -> int:",
                    "    x = 1",
                    "    def inner() -> int:",
                    "        return x",
                    "    return inner()",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = SymtableExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    partitions = _columnar_to_dicts(result.function_partition_rows)

    def _has_free_x(row: Mapping[str, object]) -> bool:
        frees = row.get("frees")
        return isinstance(frees, list) and "x" in frees

    assert any(_has_free_x(row) for row in partitions)


def test_symtable_comprehension_scope(tmp_path: Path) -> None:
    """Ensure comprehension scopes are recorded in symtable outputs."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/mod.py": "\n".join(
                [
                    "def outer() -> list[int]:",
                    "    values = [value for value in range(3)]",
                    "    return values",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = SymtableExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    scopes = _columnar_to_dicts(result.scope_rows)
    scope_types = {
        row.get("scope_type") for row in scopes if isinstance(row.get("scope_type"), str)
    }
    assert "COMPREHENSION" in scope_types
