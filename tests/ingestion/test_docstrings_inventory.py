"""Docstrings ingestion should follow the module inventory and scan profiles."""

from __future__ import annotations

from pathlib import Path

from codeintel.config import DocstringStepConfig, RepoScanStepConfig, SnapshotRef
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion.docstrings_ingest import ingest_docstrings
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.source_scanner import ScanProfile, default_code_profile
from tests._helpers.gateway import open_ingestion_gateway


def _code_profile_ignoring_dir(snapshot_repo_root: Path, ignored_dir_name: str) -> ScanProfile:
    """
    Return a ScanProfile that inherits defaults but ignores an additional directory.
    """
    base = default_code_profile(snapshot_repo_root)
    ignore_dirs = base.ignore_dirs + (ignored_dir_name,)
    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=base.include_globs,
        ignore_dirs=ignore_dirs,
        log_every=base.log_every,
        log_interval=base.log_interval,
    )


def test_docstrings_respects_scan_profile_and_module_inventory(tmp_path: Path) -> None:
    """Ensure docstrings ingest honors scan profile filters and module inventory."""
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_ignored = repo_root / "src" / "ignored"
    src_pkg.mkdir(parents=True)
    src_ignored.mkdir(parents=True)

    (src_pkg / "a.py").write_text('"""doc A"""\n', encoding="utf8")
    (src_pkg / "b.py").write_text('"""doc B"""\n', encoding="utf8")
    (src_ignored / "c.py").write_text('"""ignored doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(snapshot.repo_root)
    code_profile = _code_profile_ignoring_dir(snapshot.repo_root, "ignored")
    gateway = open_ingestion_gateway()

    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gateway,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gateway,
        cfg_docs,
        code_profile=code_profile,
    )

    rows = gateway.con.execute(
        "SELECT DISTINCT rel_path, module FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/a.py", "src/pkg/b.py"]
    assert all("/" in rel_path for rel_path in rel_paths)


def test_docstrings_uses_module_inventory_not_filesystem_scan(tmp_path: Path) -> None:
    """Verify docstrings ingest trusts core.modules instead of re-scanning the filesystem."""
    repo_root = tmp_path / "repo"
    src_pkg = repo_root / "src" / "pkg"
    src_pkg.mkdir(parents=True)

    (src_pkg / "visible.py").write_text('"""visible doc"""\n', encoding="utf8")
    (src_pkg / "ghost.py").write_text('"""ghost doc"""\n', encoding="utf8")

    snapshot = SnapshotRef(repo="demo/docstrings", commit="deadbeef", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(snapshot.repo_root)
    code_profile = default_code_profile(snapshot.repo_root)
    gateway = open_ingestion_gateway()

    cfg_scan = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(
        gateway,
        cfg=cfg_scan,
        code_profile=code_profile,
        apply_schema=True,
    )

    gateway.con.execute(
        """
        DELETE FROM core.modules
        WHERE repo = ? AND commit = ? AND path = ?
        """,
        [snapshot.repo, snapshot.commit, "src/pkg/ghost.py"],
    )

    cfg_docs = DocstringStepConfig(snapshot=snapshot)
    ingest_docstrings(
        gateway,
        cfg_docs,
        code_profile=code_profile,
    )

    rows = gateway.con.execute(
        "SELECT DISTINCT rel_path FROM core.docstrings ORDER BY rel_path"
    ).fetchall()
    rel_paths = [row[0] for row in rows]
    assert rel_paths == ["src/pkg/visible.py"]
