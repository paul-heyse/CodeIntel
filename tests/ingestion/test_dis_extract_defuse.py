"""Tests for bytecode def/use event extraction."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.dis_extract import DisExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree


def _reader_to_dicts(
    reader: pa.RecordBatchReader | pa.Table,
) -> list[dict[str, object]]:
    table = reader if isinstance(reader, pa.Table) else reader_to_table(reader)
    return list(table.to_pylist())


def test_dis_extract_defuse_events(tmp_path: Path) -> None:
    """Ensure def/use extraction emits basic event rows."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/defs.py": "\n".join(
                [
                    "def compute() -> int:",
                    "    x = 1",
                    "    y = x + 2",
                    "    return y",
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    step = DisExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    rows = _reader_to_dicts(result.defuse_event_rows_reader)
    kinds = {row.get("event_kind") for row in rows if isinstance(row.get("event_kind"), str)}
    names = {row.get("name") for row in rows if isinstance(row.get("name"), str)}
    assert "DEF" in kinds
    assert "USE" in kinds
    assert "x" in names
