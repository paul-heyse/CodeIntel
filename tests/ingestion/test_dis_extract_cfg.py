"""Tests for bytecode CFG extraction."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

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
    assert _reader_row_count(result.block_rows_reader) > 0
    assert _reader_row_count(result.cfg_edge_rows_reader) > 0


def _reader_to_dicts(reader: pa.RecordBatchReader) -> list[dict[str, object]]:
    """Convert a RecordBatchReader into row dictionaries for assertions.

    Returns
    -------
    list[dict[str, object]]
        Rows converted from the reader stream.
    """
    table = pa.Table.from_batches(reader, schema=reader.schema)
    return list(table.to_pylist())


def _reader_row_count(reader: pa.RecordBatchReader) -> int:
    """Count rows in a RecordBatchReader without materializing a table.

    Returns
    -------
    int
        Total number of rows across batches.
    """
    return sum(batch.num_rows for batch in reader)


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

    exception_rows = _reader_to_dicts(result.exception_rows_reader)
    assert exception_rows
    cfg_rows = _reader_to_dicts(result.cfg_edge_rows_reader)
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

    block_rows = _reader_to_dicts(result.block_rows_reader)
    assert block_rows
    for row in block_rows:
        start_offset = row.get("start_offset")
        end_offset = row.get("end_offset")
        if isinstance(start_offset, int) and isinstance(end_offset, int):
            assert start_offset < end_offset
            start_label = row.get("start_label")
            assert start_label == f"L{start_offset}"


def test_dis_extract_match_async_cfg(tmp_path: Path) -> None:
    """Ensure match/case and async control flow produce CFG rows."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/flow.py": "\n".join(
                [
                    "import asyncio",
                    "",
                    "def classify(value: int) -> str:",
                    "    match value:",
                    "        case 0:",
                    '            return "zero"',
                    "        case 1 | 2:",
                    '            return "small"',
                    "        case _:",
                    '            return "other"',
                    "",
                    "async def handle(value: int) -> int:",
                    "    if value > 0:",
                    "        return value",
                    "    await asyncio.sleep(0)",
                    "    return value",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = DisExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success
    assert _reader_row_count(result.block_rows_reader) > 0
    assert _reader_row_count(result.cfg_edge_rows_reader) > 0
