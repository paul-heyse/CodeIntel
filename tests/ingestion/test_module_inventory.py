"""Guardrails for module inventory and SourceScanner usage."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest

from codeintel.config import SnapshotRef
from codeintel.ingestion import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    RepoScanStep,
)
from codeintel.ingestion.common import iter_modules
from codeintel.ingestion.infrastructure_utilities.source_scanner import default_code_profile
from codeintel.ingestion.steps import (
    ast_extract,
    config_ingest,
    cst_extract,
    docstrings_extract,
    repo_scan,
    typing_ingest,
)
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
    """Ensure SourceScanner only appears in repo_scan and config_ingest step modules."""
    allowed = {
        "codeintel.ingestion.steps.repo_scan",
        "codeintel.ingestion.steps.config_ingest",
    }
    modules = {
        "codeintel.ingestion.steps.repo_scan": repo_scan,
        "codeintel.ingestion.steps.config_ingest": config_ingest,
        "codeintel.ingestion.steps.docstrings_extract": docstrings_extract,
        "codeintel.ingestion.steps.typing_ingest": typing_ingest,
        "codeintel.ingestion.steps.ast_extract": ast_extract,
        "codeintel.ingestion.steps.cst_extract": cst_extract,
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
    gateway = open_ingestion_gateway()
    profile = default_code_profile(snapshot.repo_root)

    # Use Step-based API
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(snapshot.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)

    scan_step = RepoScanStep(
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
    )
    _, modules, _ = scan_step.execute(
        repo=snapshot.repo,
        commit=snapshot.commit,
        repo_root=snapshot.repo_root,
        profile=profile,
    )

    module_map = load_module_map(
        gateway,
        snapshot.repo,
        snapshot.commit,
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

    scan_paths = sorted(module.rel_path for module in modules)
    if scan_paths != rel_paths:
        pytest.fail(
            f"Scan modules {scan_paths} differ from module_map derived paths {rel_paths}"
        )
