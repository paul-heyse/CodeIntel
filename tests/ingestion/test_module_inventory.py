"""Guardrails for module inventory and SourceScanner usage."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.adapters import FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import (
    ast_extract,
    config_ingest,
    cst_extract,
    docstrings_extract,
    repo_scan,
    typing_ingest,
)
from codeintel.storage.helpers.module_index import load_module_map
from tests._helpers import modules_expected_from_repo_tree
from tests._helpers.assertions import (
    MissingExtraOptions,
    ModuleMapDiffOptions,
    format_missing_extra,
    format_module_map_diff,
    module_map_from_path_map,
)
from tests._helpers.ingestion import materialize_repo_scan_result, module_inventory_context

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
    with module_inventory_context(tmp_path) as ctx:
        scan_result = ctx.scan_step.execute(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            repo_root=ctx.snapshot.repo_root,
            profile=ctx.profile,
        )
        materialize_repo_scan_result(
            ctx.gateway,
            scan_result,
            snapshot=ctx.snapshot,
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
        expected_map = modules_expected_from_repo_tree(ctx.snapshot.repo_root)
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

        expected_module_map = module_map_from_path_map(expected_map)
        actual_module_map = module_map_from_path_map(module_map)
        if actual_module_map != expected_module_map:
            pytest.fail(
                format_module_map_diff(
                    expected_module_map,
                    actual_module_map,
                    options=ModuleMapDiffOptions(context="core.modules"),
                )
            )

        scan_paths = sorted(module.rel_path for module in scan_result.modules)
        if scan_paths != rel_paths:
            pytest.fail(
                f"Scan modules {scan_paths} differ from module_map derived paths {rel_paths}"
            )
