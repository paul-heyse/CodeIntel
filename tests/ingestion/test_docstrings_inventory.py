"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from tests._helpers.assertions import (
    MissingExtraOptions,
    ModulesAssertions,
    assert_target_ok,
    format_missing_extra,
)
from tests._helpers.fixtures.repos import write_tree
from tests._helpers.fixtures.rows import RepoMapRow, insert_rows
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)

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
        ModulesAssertions(harness.ctx.gateway, snapshot).inventory_consistent()

        rows = harness.ctx.gateway.con.execute(
            "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
        ).fetchall()
        rel_paths = [row[0] for row in rows]
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

        snapshot = harness.ctx.snapshot
        gateway = harness.ctx.gateway
        gateway.con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND path = ?",
            [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
        )
        gateway.con.execute(
            "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
            [snapshot.repo, snapshot.commit],
        )
        rows = gateway.con.execute(
            "SELECT module, path FROM core.modules WHERE repo = ? AND commit = ? ORDER BY module",
            [snapshot.repo, snapshot.commit],
        ).fetchall()
        modules = {row[0]: row[1] for row in rows}
        insert_rows(
            gateway,
            [
                RepoMapRow(
                    repo=snapshot.repo,
                    commit=snapshot.commit,
                    modules=modules,
                )
            ],
        )
        ModulesAssertions(gateway, snapshot).inventory_consistent()

        docstrings_result = harness.run_targets(["docstrings"])
        docstrings_record = harness.record("docstrings", result=docstrings_result)
        assert_target_ok(docstrings_record)

        rows = gateway.con.execute(
            "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
        ).fetchall()
        rel_paths = [row[0] for row in rows]
        expected_paths = ["src/pkg/visible.py"]
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
