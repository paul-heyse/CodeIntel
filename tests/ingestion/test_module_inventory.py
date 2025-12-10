"""Guardrails for module inventory and SourceScanner usage."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest

from codeintel.ingestion import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import (
    ast_extract,
    config_ingest,
    cst_extract,
    docstrings_extract,
    repo_scan,
    typing_ingest,
)
from codeintel.storage.helpers.module_index import load_module_map
from tests._helpers.ingestion import module_inventory_context


def _source(module: ModuleType) -> str:
    return inspect.getsource(module)


def test_scanning_only_used_in_repo_scan_and_config_ingest() -> None:
    """Ensure SourceScanner only appears in repo_scan and config_ingest compute modules."""
    allowed = {
        "codeintel.ingestion.compute.repo_scan",
        "codeintel.ingestion.compute.config_ingest",
    }
    modules = {
        "codeintel.ingestion.compute.repo_scan": repo_scan,
        "codeintel.ingestion.compute.config_ingest": config_ingest,
        "codeintel.ingestion.compute.docstrings_extract": docstrings_extract,
        "codeintel.ingestion.compute.typing_ingest": typing_ingest,
        "codeintel.ingestion.compute.ast_extract": ast_extract,
        "codeintel.ingestion.compute.cst_extract": cst_extract,
    }

    offenders: list[str] = []
    for name, module in modules.items():
        if "SourceScanner(" in _source(module) and name not in allowed:
            offenders.append(name)

    if offenders:
        pytest.fail(f"SourceScanner used outside allowed modules: {offenders}")


def test_module_inventory_round_trip(tmp_path: Path) -> None:
    """Verify module inventory round-trips through core.modules and iter_modules."""
    ctx = module_inventory_context(tmp_path)
    _, modules, _ = ctx.scan_step.execute(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        repo_root=ctx.snapshot.repo_root,
        profile=ctx.profile,
    )

    module_map = load_module_map(
        ctx.gateway,
        ctx.snapshot.repo,
        ctx.snapshot.commit,
        language="python",
        logger=None,
    )
    records = list(
        FilesystemDiscoveryAdapter.iter_modules(
            module_map, ctx.snapshot.repo_root, logger=None, scan_profile=ctx.profile
        )
    )

    rel_paths = sorted(record.rel_path for record in records)
    expected = ["src/pkg/a.py", "src/pkg/b.py"]
    if rel_paths != expected:
        pytest.fail(f"Unexpected module paths {rel_paths}, expected {expected}")
    if not all("/" in rel_path for rel_path in rel_paths):
        pytest.fail(f"Non-POSIX module paths: {rel_paths}")

    scan_paths = sorted(module.rel_path for module in modules)
    if scan_paths != rel_paths:
        pytest.fail(f"Scan modules {scan_paths} differ from module_map derived paths {rel_paths}")
