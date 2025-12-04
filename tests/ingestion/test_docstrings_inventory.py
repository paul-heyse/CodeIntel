"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config import SnapshotRef
from codeintel.ingestion import (
    DocstringsExtractStep,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    RepoScanStep,
)
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
)
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway


def _code_profile_ignoring_dir(snapshot_repo_root: Path, ignored_dir_name: str) -> ScanProfile:
    """Build a scan profile that ignores an extra directory.

    Returns
    -------
    ScanProfile
        Profile that mirrors the default code profile but skips the provided directory name.
    """
    base = default_code_profile(snapshot_repo_root)
    ignore_dirs = (*base.ignore_dirs, ignored_dir_name)
    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=base.include_globs,
        ignore_dirs=ignore_dirs,
        log_every=base.log_every,
        log_interval=base.log_interval,
    )


def _create_scan_steps(
    gateway: StorageGateway,
    repo_root: Path,
) -> tuple[RepoScanStep, DocstringsExtractStep]:
    """Create scan and docstring steps from gateway and repo root.

    Returns
    -------
    tuple[RepoScanStep, DocstringsExtractStep]
        Scan step and docstrings extraction step.
    """
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    scan_step = RepoScanStep(
        storage=storage, discovery=discovery, change_detection=change_detection
    )
    doc_step = DocstringsExtractStep(storage=storage, discovery=discovery)
    return scan_step, doc_step


def _scan_and_extract_docstrings(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    profile: ScanProfile,
    module_filter: Iterable[ModuleRecord] | None = None,
) -> list[str]:
    """Scan modules and extract docstrings, returning list of rel_paths.

    Returns
    -------
    list[str]
        List of relative paths where docstrings were extracted.
    """
    scan_step, doc_step = _create_scan_steps(gateway, snapshot.repo_root)

    _, modules, _ = scan_step.execute(
        repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root, profile=profile
    )

    target_modules = list(module_filter) if module_filter is not None else list(modules)
    doc_step.execute(target_modules, repo=snapshot.repo, commit=snapshot.commit)

    rows = gateway.con.execute(
        "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    return [row[0] for row in rows]


def test_docstrings_respects_scan_profile_and_module_inventory(tmp_path: Path) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "pkg").mkdir(parents=True)
    (repo_root / "src" / "ignored").mkdir(parents=True)

    (repo_root / "src" / "pkg" / "a.py").write_text('"""doc A"""\n', encoding="utf8")
    (repo_root / "src" / "pkg" / "b.py").write_text('"""doc B"""\n', encoding="utf8")
    (repo_root / "src" / "ignored" / "c.py").write_text('"""ignored doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="abc123", repo_root=repo_root)
    code_profile = _code_profile_ignoring_dir(snapshot.repo_root, "ignored")
    gateway = open_ingestion_gateway()

    rel_paths = _scan_and_extract_docstrings(gateway, snapshot, code_profile)
    expected_paths = ["src/pkg/a.py", "src/pkg/b.py"]

    if rel_paths != expected_paths:
        pytest.fail(f"Unexpected docstring paths {rel_paths}, expected {expected_paths}")
    if not all("/" in rel_path for rel_path in rel_paths):
        pytest.fail(f"Non-POSIX paths observed: {rel_paths}")


def test_docstrings_uses_module_inventory_not_filesystem_scan(tmp_path: Path) -> None:
    """Verify docstrings ingest trusts core.modules instead of re-scanning the filesystem."""
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_pkg.mkdir(parents=True)

    (src_pkg / "visible.py").write_text('"""visible doc"""\n', encoding="utf8")
    (src_pkg / "ghost.py").write_text('"""ghost doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="deadbeef", repo_root=repo_root)
    code_profile = default_code_profile(snapshot.repo_root)
    gateway = open_ingestion_gateway()

    scan_step, doc_step = _create_scan_steps(gateway, repo_root)
    _, modules, _ = scan_step.execute(
        repo=snapshot.repo, commit=snapshot.commit, repo_root=repo_root, profile=code_profile
    )

    # Delete ghost.py from inventory and filter from modules
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND path = ?",
        [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
    )
    filtered_modules = [m for m in modules if m.rel_path != "src/pkg/ghost.py"]

    doc_step.execute(filtered_modules, repo=snapshot.repo, commit=snapshot.commit)

    rows = gateway.con.execute(
        "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    if rel_paths != ["src/pkg/visible.py"]:
        pytest.fail(f"Docstrings ingested for unexpected paths: {rel_paths}")
