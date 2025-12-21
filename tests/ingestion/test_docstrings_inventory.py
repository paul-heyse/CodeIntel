"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from tests._helpers.factories import make_snapshot
from tests._helpers.ingestion import (
    build_repo_with_variants,
    build_scan_profile,
    create_scan_step,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


@contextmanager
def _docstrings_setup(
    tmp_path: Path,
    *,
    gateway: StorageGateway,
    repo_structure: dict[str, str],
    ignore_dirs: tuple[str, ...] = (),
) -> Iterator[SimpleNamespace]:
    repo_root = build_repo_with_variants(tmp_path, extra_structure=repo_structure)
    profile = build_scan_profile(repo_root, ignore_dirs=ignore_dirs)
    scan_step, storage, discovery = create_scan_step(gateway, repo_root, tmp_path)
    ctx = SimpleNamespace(
        repo_root=repo_root,
        gateway=gateway,
        profile=profile,
        scan_step=scan_step,
        storage=storage,
        discovery=discovery,
    )
    yield ctx


def test_docstrings_respects_scan_profile_and_module_inventory(
    tmp_path: Path, ingestion_gateway: StorageGateway
) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    structure = {
        "src/pkg/a.py": '"""doc A"""\n',
        "src/pkg/b.py": '"""doc B"""\n',
        "src/ignored/c.py": '"""ignored doc"""\n',
    }
    with _docstrings_setup(
        tmp_path,
        gateway=ingestion_gateway,
        repo_structure=structure,
        ignore_dirs=("ignored",),
    ) as setup:
        snapshot = make_snapshot(repo="demo/docstrings", commit="abc123", repo_root=setup.repo_root)
        doc_step = DocstringsExtractStep(storage=setup.storage, discovery=setup.discovery)

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


def test_docstrings_uses_module_inventory_not_filesystem_scan(
    tmp_path: Path, ingestion_gateway: StorageGateway
) -> None:
    """Verify docstrings ingest trusts core.modules instead of re-scanning the filesystem."""
    structure = {
        "src/pkg/visible.py": '"""visible doc"""\n',
        "src/pkg/ghost.py": '"""ghost doc"""\n',
    }
    with _docstrings_setup(
        tmp_path,
        gateway=ingestion_gateway,
        repo_structure=structure,
    ) as setup:
        snapshot = make_snapshot(repo="demo/docstrings", repo_root=setup.repo_root)
        doc_step = DocstringsExtractStep(storage=setup.storage, discovery=setup.discovery)
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
