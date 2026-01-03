"""Guardrails for module inventory and SourceScanner usage."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.compute import (
    ast_extract,
    config_ingest,
    cst_extract,
    docstrings_extract,
    repo_scan,
    typing_ingest,
)
from tests._helpers import modules_expected_from_repo_tree
from tests._helpers.assertions import (
    MissingExtraOptions,
    ModuleMapDiffOptions,
    format_missing_extra,
    format_module_map_diff,
    module_map_from_path_map,
)
from tests._helpers.ingestion import build_parquet_repo_scan_context
from tests._helpers.parquet_datasets import read_snapshot_rows

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


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
    ctx = build_parquet_repo_scan_context(tmp_path)
    rows = read_snapshot_rows(
        ctx.dataset_root,
        table_key="core.modules",
        snapshot_id=ctx.snapshot.commit,
        columns=("module", "path"),
    )
    rel_paths: list[str] = []
    for row in rows:
        path_value = row.get("path")
        if isinstance(path_value, str):
            rel_paths.append(path_value)
    rel_paths.sort()
    expected_map = modules_expected_from_repo_tree(ctx.repo_root)
    expected_paths = sorted(expected_map)
    if rel_paths != expected_paths:
        pytest.fail(
            format_missing_extra(
                expected_paths,
                rel_paths,
                options=MissingExtraOptions(
                    noun="module paths",
                    context="module inventory",
                ),
            )
        )
    if not all("/" in rel_path for rel_path in rel_paths):
        pytest.fail(f"Non-POSIX module paths: {rel_paths}")

    module_path_map: dict[str, str] = {}
    for row in rows:
        module_value = row.get("module")
        path_value = row.get("path")
        if isinstance(module_value, str) and isinstance(path_value, str):
            module_path_map[path_value] = module_value
    expected_module_map = module_map_from_path_map(expected_map)
    actual_module_map = module_map_from_path_map(module_path_map)
    if actual_module_map != expected_module_map:
        pytest.fail(
            format_module_map_diff(
                expected_module_map,
                actual_module_map,
                options=ModuleMapDiffOptions(context="core.modules parquet"),
            )
        )

    scan_paths = sorted(module.rel_path for module in ctx.scan_result.modules)
    if scan_paths != rel_paths:
        pytest.fail(f"Scan modules {scan_paths} differ from dataset-derived paths {rel_paths}")
