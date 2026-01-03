"""Tests for inspect overlay extraction."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.native.options.ingestion import InspectExtractOptions
from codeintel.core.columnar.rows import ColumnarRows
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.inspect_extract import InspectExtractStep
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


def test_inspect_overlay_basic(tmp_path: Path) -> None:
    """Ensure inspect overlay extracts module metadata."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/inspectable.py": "\n".join(
                [
                    "def greet(name: str) -> str:",
                    '    return f"hi {name}"',
                ]
            ),
        },
    )
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    options = InspectExtractOptions(
        enable=True,
        module_allowlist=["pkg.inspectable"],
        use_subprocess=False,
        max_objects=250,
    )
    step = InspectExtractStep(FilesystemDiscoveryAdapter(repo_root), options=options)
    result = step.execute(modules, repo="demo", commit="abc123")
    assert result.result.success

    rows = _columnar_to_dicts(result.object_rows)
    assert any(row.get("module_name") == "pkg.inspectable" for row in rows)
