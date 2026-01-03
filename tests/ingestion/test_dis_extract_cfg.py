"""Tests for bytecode CFG extraction."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.columnar.rows import ColumnarRows, columnar_row_count
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.dis_extract import DisExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree


def test_dis_extract_cfg_edges(tmp_path: Path) -> None:
    """Ensure CFG extraction emits block and edge rows."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/flow.py": "\n".join(
                [
                    "def flow(x: int) -> int:",
                    "    if x > 0:",
                    "        return x",
                    "    total = 0",
                    "    for i in range(3):",
                    "        total += i",
                    "    return total",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = DisExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success
    assert columnar_row_count(result.block_rows) > 0
    assert columnar_row_count(result.cfg_edge_rows) > 0


def _columnar_to_dicts(rows: ColumnarRows) -> list[dict[str, object]]:
    """Convert columnar rows into a list of dicts for assertions.

    Returns
    -------
    list[dict[str, object]]
        Rows converted from columnar storage.
    """
    if not rows:
        return []
    columns = list(rows.keys())
    if not columns:
        return []
    row_count = len(rows[columns[0]])
    return [{col: rows[col][idx] for col in columns} for idx in range(row_count)]


def test_dis_extract_exception_table_edges(tmp_path: Path) -> None:
    """Ensure exception table parsing emits exception edges."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/flow.py": "\n".join(
                [
                    "def handle(path: str) -> str:",
                    "    try:",
                    "        with open(path) as handle:",
                    "            return handle.read()",
                    "    except OSError:",
                    '        return ""',
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = DisExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    exception_rows = _columnar_to_dicts(result.exception_rows)
    assert exception_rows
    cfg_rows = _columnar_to_dicts(result.cfg_edge_rows)
    kinds = {row.get("kind") for row in cfg_rows if isinstance(row.get("kind"), str)}
    assert "EXCEPTION" in kinds


def test_dis_extract_block_boundaries(tmp_path: Path) -> None:
    """Ensure block boundaries align with instruction offsets."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/blocks.py": "\n".join(
                [
                    "def compute(values: list[int]) -> int:",
                    "    total = 0",
                    "    for value in values:",
                    "        if value % 2:",
                    "            total += value",
                    "    return total",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = DisExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    block_rows = _columnar_to_dicts(result.block_rows)
    assert block_rows
    for row in block_rows:
        start_offset = row.get("start_offset")
        end_offset = row.get("end_offset")
        if isinstance(start_offset, int) and isinstance(end_offset, int):
            assert start_offset < end_offset
            start_label = row.get("start_label")
            assert start_label == f"L{start_offset}"
