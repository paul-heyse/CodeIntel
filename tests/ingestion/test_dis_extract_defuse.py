"""Tests for bytecode def/use event extraction."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.columnar.rows import ColumnarRows
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.dis_extract import DisExtractStep
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

    rows = _columnar_to_dicts(result.defuse_event_rows)
    kinds = {row.get("event_kind") for row in rows if isinstance(row.get("event_kind"), str)}
    names = {row.get("name") for row in rows if isinstance(row.get("name"), str)}
    assert "DEF" in kinds
    assert "USE" in kinds
    assert "x" in names
