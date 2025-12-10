"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion import DocstringsExtractStep
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import ScanSetupOptions, make_scan_setup


def test_docstrings_respects_scan_profile_and_module_inventory(tmp_path: Path) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "src/pkg/a.py": '"""doc A"""\n',
                "src/pkg/b.py": '"""doc B"""\n',
                "src/ignored/c.py": '"""ignored doc"""\n',
            },
            ignore_dirs=("ignored",),
            gateway_factory=GatewayFactory(),
        ),
    )
    snapshot = make_snapshot(repo="demo/docstrings", commit="abc123", repo_root=setup.repo_root)
    doc_step = DocstringsExtractStep(storage=setup.storage, discovery=setup.discovery)

    try:
        _, modules, _ = setup.scan_step.execute(
            repo=snapshot.repo,
            commit=snapshot.commit,
            repo_root=snapshot.repo_root,
            profile=setup.profile,
        )
        doc_step.execute(list(modules), repo=snapshot.repo, commit=snapshot.commit)
        rows = setup.gateway.con.execute(
            "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
        ).fetchall()
        rel_paths = [row[0] for row in rows]
        expected_paths = ["src/pkg/a.py", "src/pkg/b.py"]

        if rel_paths != expected_paths:
            pytest.fail(f"Unexpected docstring paths {rel_paths}, expected {expected_paths}")
        if not all("/" in rel_path for rel_path in rel_paths):
            pytest.fail(f"Non-POSIX paths observed: {rel_paths}")
    finally:
        setup.gateway.close()


def test_docstrings_uses_module_inventory_not_filesystem_scan(tmp_path: Path) -> None:
    """Verify docstrings ingest trusts core.modules instead of re-scanning the filesystem."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "src/pkg/visible.py": '"""visible doc"""\n',
                "src/pkg/ghost.py": '"""ghost doc"""\n',
            },
            gateway_factory=GatewayFactory(),
        ),
    )
    snapshot = make_snapshot(repo="demo/docstrings", repo_root=setup.repo_root)
    doc_step = DocstringsExtractStep(storage=setup.storage, discovery=setup.discovery)
    try:
        _, modules, _ = setup.scan_step.execute(
            repo=snapshot.repo,
            commit=snapshot.commit,
            repo_root=setup.repo_root,
            profile=setup.profile,
        )

        setup.gateway.con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND path = ?",
            [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
        )
        filtered_modules = [m for m in modules if m.rel_path != "src/pkg/ghost.py"]

        doc_step.execute(filtered_modules, repo=snapshot.repo, commit=snapshot.commit)

        rows = setup.gateway.con.execute(
            "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
        ).fetchall()
        rel_paths = [row[0] for row in rows]
        if rel_paths != ["src/pkg/visible.py"]:
            pytest.fail(f"Docstrings ingested for unexpected paths: {rel_paths}")
    finally:
        setup.gateway.close()
