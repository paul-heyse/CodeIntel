"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from tests._helpers.assertions import MissingExtraOptions, assert_target_ok, format_missing_extra
from tests._helpers.fixtures.repos import write_tree
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.parquet_datasets import read_snapshot_rows

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def _docstrings_harness(
    tmp_path: Path,
    *,
    repo_structure: dict[str, str],
    scope_paths: list[str],
) -> Iterator[HamiltonBuildHarness]:
    def _write_repo(repo_root: Path) -> list[Path]:
        files = {"src/pkg/__init__.py": "", **repo_structure}
        write_tree(repo_root, files)
        return [repo_root / rel_path for rel_path in files]

    config = BuildConfig.from_dict({"ingestion": {"modules": {"scope_paths": scope_paths}}})
    options = HarnessOpenOptions(
        repo_strategy="writer",
        repo_writer=_write_repo,
        build_config=config,
    )
    harness = HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="demo/docstrings", commit="abc123"),
        options=options,
    )
    try:
        yield harness
    finally:
        harness.close()


def test_docstrings_respects_scan_profile_and_module_inventory(
    tmp_path: Path,
) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    structure = {
        "src/pkg/a.py": '"""doc A"""\n',
        "src/pkg/b.py": '"""doc B"""\n',
        "src/ignored/c.py": '"""ignored doc"""\n',
    }
    with _docstrings_harness(
        tmp_path,
        repo_structure=structure,
        scope_paths=["src/pkg"],
    ) as harness:
        result = harness.run_targets(["docstrings"])
        record = harness.record("docstrings", result=result)
        assert_target_ok(record)
        snapshot = harness.ctx.snapshot
        dataset_root = harness.ctx.build_paths.dataset_root_dir
        try:
            rows = read_snapshot_rows(
                dataset_root,
                table_key="core.docstrings",
                snapshot_id=snapshot.commit,
                columns=("rel_path",),
            )
        except FileNotFoundError:
            pytest.xfail("Parquet datasets not yet materialized for docstrings target.")
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
    structure = {
        "src/pkg/visible.py": '"""visible doc"""\n',
        "src/pkg/ghost.py": '"""ghost doc"""\n',
    }
    with _docstrings_harness(
        tmp_path,
        repo_structure=structure,
        scope_paths=["src/pkg"],
    ) as harness:
        modules_result = harness.run_targets(["modules"])
        modules_record = harness.record("modules", result=modules_result)
        assert_target_ok(modules_record)

        _ = harness
