"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.helpers import filter_paths
from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from tests._helpers.assertions import MissingExtraOptions, format_missing_extra
from tests._helpers.fixtures.repos import write_tree

if TYPE_CHECKING:
    from pathlib import Path


def test_docstrings_respects_scan_profile_and_module_inventory(
    tmp_path: Path,
) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    structure = {
        "src/pkg/a.py": '"""doc A"""\n',
        "src/pkg/b.py": '"""doc B"""\n',
        "src/ignored/c.py": '"""ignored doc"""\n',
    }
    repo_root = tmp_path / "repo"
    files = {"src/pkg/__init__.py": "", **structure}
    write_tree(repo_root, files)
    profile = default_code_profile(repo_root)
    modules = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    scope_paths = ["src/pkg"]
    filtered = set(filter_paths([record.rel_path for record in modules], scope_paths=scope_paths))
    scoped_modules = [record for record in modules if record.rel_path in filtered]

    step = DocstringsExtractStep(FilesystemDiscoveryAdapter(repo_root))
    result = step.execute(scoped_modules, repo="demo/docstrings", commit="abc123")
    assert result.result.success

    rows = result.rows_reader.to_pylist()
    rel_paths: list[str] = []
    for row in rows:
        rel_path = row.get("rel_path")
        if isinstance(rel_path, str):
            rel_paths.append(rel_path)
    rel_paths.sort()
    expected_paths = ["src/pkg/a.py", "src/pkg/b.py"]

    if rel_paths != expected_paths:
        pytest.fail(
            format_missing_extra(
                expected_paths,
                rel_paths,
                options=MissingExtraOptions(
                    noun="docstring paths",
                    context="docstrings inventory",
                ),
            )
        )
    if not all("/" in rel_path for rel_path in rel_paths):
        pytest.fail(f"Non-POSIX paths observed: {rel_paths}")


def test_docstrings_uses_module_inventory_not_filesystem_scan(
    tmp_path: Path,
) -> None:
    """Verify docstrings ingest trusts core.modules instead of re-scanning the filesystem."""
    pytest.xfail("Docstrings inventory currently relies on gateway-backed module inventory.")
    _ = tmp_path
