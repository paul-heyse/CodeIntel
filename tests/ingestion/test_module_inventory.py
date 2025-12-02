"""Guardrails for module inventory and SourceScanner usage."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest
from codeintel.ingestion.repo_scan import ingest_repo

from codeintel.config import RepoScanStepConfig, SnapshotRef
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion import (
    config_ingest,
    cst_extract,
    docstrings_ingest,
    py_ast_extract,
    repo_scan,
    typing_ingest,
)
from codeintel.ingestion.common import iter_modules
from codeintel.ingestion.infrastructure_utilities.source_scanner import default_code_profile
from codeintel.storage.module_index import load_module_map
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway


def _source(module: ModuleType) -> str:
    return inspect.getsource(module)


def _make_snapshot(tmp_path: Path) -> SnapshotRef:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")
    (src_dir / "b.py").write_text("print('b')\n", encoding="utf8")
    return SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)


def test_source_scanner_only_used_in_repo_scan_and_config_ingest() -> None:
    """Ensure SourceScanner only appears in repo_scan and config_ingest modules."""
    allowed = {
        "codeintel.ingestion.repo_scan",
        "codeintel.ingestion.config_ingest",
    }
    modules = {
        "codeintel.ingestion.repo_scan": repo_scan,
        "codeintel.ingestion.config_ingest": config_ingest,
        "codeintel.ingestion.docstrings_ingest": docstrings_ingest,
        "codeintel.ingestion.typing_ingest": typing_ingest,
        "codeintel.ingestion.py_ast_extract": py_ast_extract,
        "codeintel.ingestion.cst_extract": cst_extract,
    }

    offenders: list[str] = []
    for name, module in modules.items():
        if "SourceScanner(" in _source(module) and name not in allowed:
            offenders.append(name)

    if offenders:
        pytest.fail(f"SourceScanner used outside allowed modules: {offenders}")


def test_module_inventory_round_trip(tmp_path: Path) -> None:
    """Verify module inventory round-trips through core.modules and iter_modules."""
    snapshot = _make_snapshot(tmp_path)
    paths = BuildPaths.from_repo_root(snapshot.repo_root)
    gateway = open_ingestion_gateway()

    cfg = RepoScanStepConfig(snapshot=snapshot, paths=paths, tool_runner=None)
    profile = default_code_profile(snapshot.repo_root)

    tracker = ingest_repo(
        gateway,
        cfg=cfg,
        code_profile=profile,
        apply_schema=True,
    )

    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=None,
    )
    records = list(iter_modules(module_map, snapshot.repo_root, logger=None, scan_profile=profile))

    rel_paths = sorted(record.rel_path for record in records)
    expected = ["src/pkg/a.py", "src/pkg/b.py"]
    if rel_paths != expected:
        pytest.fail(f"Unexpected module paths {rel_paths}, expected {expected}")
    if not all("/" in rel_path for rel_path in rel_paths):
        pytest.fail(f"Non-POSIX module paths: {rel_paths}")

    tracker_paths = sorted(module.rel_path for module in tracker.modules)
    if tracker_paths != rel_paths:
        pytest.fail(
            f"Tracker modules {tracker_paths} differ from module_map derived paths {rel_paths}"
        )
