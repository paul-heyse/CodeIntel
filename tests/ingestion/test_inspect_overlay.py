"""Tests for inspect overlay extraction."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from codeintel.build.hamilton.native.options.ingestion import InspectExtractOptions
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.inspect_extract import InspectExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.fixtures.repos import write_tree


def _reader_to_dicts(reader: pa.RecordBatchReader) -> list[dict[str, object]]:
    table = pa.Table.from_batches(reader, schema=reader.schema)
    return list(table.to_pylist())


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

    rows = _reader_to_dicts(result.object_rows_reader)
    assert any(row.get("module_name") == "pkg.inspectable" for row in rows)


def test_inspect_overlay_wrapped_callable(tmp_path: Path) -> None:
    """Ensure unwrap hops capture decorated callables."""
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {
            "pkg/inspectable.py": "\n".join(
                [
                    "import functools",
                    "",
                    "def decorator(func):",
                    "    @functools.wraps(func)",
                    "    def wrapper(*args, **kwargs):",
                    "        return func(*args, **kwargs)",
                    "    return wrapper",
                    "",
                    "@decorator",
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

    unwrap_rows = _reader_to_dicts(result.unwrap_rows_reader)
    assert any(row.get("has_wrapped") is True for row in unwrap_rows)
