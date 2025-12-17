"""Native Hamilton implementation for modules target.

This module implements repository scanning as a native Hamilton pipeline with:
- t__modules__scan: Execute RepoScanStep to discover modules
- t__modules__write_repo_map: Persist repo_map entry
- t__modules: Materialize with validators and return TargetRunRecord

Phase 2: Ingestion domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hamilton.function_modifiers import cache, tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import build_scan_profile, filter_modules
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.change_detection import ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ModuleScanResult:
    """Result from repository module scanning.

    Attributes
    ----------
    success
        Whether the scan completed successfully.
    modules
        Discovered module records.
    change_set
        Computed change set for incremental processing.
    table_counts
        Row counts per produced table.
    error
        Error message if scan failed.
    """

    success: bool
    modules: Sequence[ModuleRecord] = field(default_factory=tuple)
    change_set: ChangeSet | None = None
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class RepoMapWriteResult:
    """Result from writing repo_map entry.

    Attributes
    ----------
    success
        Whether the write completed successfully.
    row_count
        Number of rows written (typically 1 for repo_map).
    error
        Error message if write failed.
    """

    success: bool
    row_count: int = 0
    error: str | None = None


@cache(format="memory")
@tag(domain="ingestion", target="modules", node_type="compute")
def t__modules__scan(env: BuildEnv) -> ModuleScanResult:
    """Execute repository scan to discover modules.

    This is the primary compute node for the modules target. It scans
    the repository tree, discovers Python modules, computes file hashes
    for change detection, and persists the modules table.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.

    Returns
    -------
    ModuleScanResult
        Result containing discovered modules and row counts.

    Notes
    -----
    The modules target is unique in that it has no upstream dependencies.
    It is the root of the ingestion domain dependency tree.
    """
    try:
        storage = DuckDBStorageAdapter(env.gateway)
        discovery = FilesystemDiscoveryAdapter(env.snapshot.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        # Build scan profile from config
        opts = ModuleIngestOptions()
        profile = build_scan_profile(env.snapshot.repo_root, opts)

        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
            module_filter=lambda discovered: filter_modules(discovered, opts),
        )

        result, modules, change_set = step.execute(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
            profile=profile,
            full_rebuild=False,
        )

        return ModuleScanResult(
            success=True,
            modules=tuple(modules),
            change_set=change_set,
            table_counts=result.table_counts or {},
        )

    except Exception as exc:
        log.exception("Module scan failed")
        return ModuleScanResult(
            success=False,
            error=str(exc),
        )


@tag(domain="ingestion", target="modules", node_type="compute")
def t__modules__write_repo_map(
    env: BuildEnv,
    t__modules__scan: ModuleScanResult,
) -> RepoMapWriteResult:
    """Write repo_map entry for the repository snapshot.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules__scan
        Upstream scan result containing discovered modules.

    Returns
    -------
    RepoMapWriteResult
        Result indicating success/failure of repo_map write.
    """
    if not t__modules__scan.success:
        return RepoMapWriteResult(
            success=False,
            error=f"Upstream scan failed: {t__modules__scan.error}",
        )

    try:
        modules = t__modules__scan.modules
        generated_at = datetime.now(tz=UTC).isoformat()

        # Build module entries JSON
        module_entries: dict[str, str] = {}
        for module in modules:
            name = getattr(module, "module_name", None) or getattr(module, "name", None)
            rel_path = getattr(module, "rel_path", None) or getattr(module, "path", None)
            if name is None:
                name = str(module)
            module_entries[str(name)] = str(rel_path) if rel_path is not None else ""

        modules_json = json.dumps(module_entries)
        overlays_json = json.dumps({})

        warehouse = Warehouse(env.gateway)
        warehouse.materialize_mappings(
            "core.repo_map",
            [
                {
                    "repo": env.snapshot.repo,
                    "commit": env.snapshot.commit,
                    "modules": modules_json,
                    "overlays": overlays_json,
                    "generated_at": generated_at,
                }
            ],
            options=MaterializeOptions(snapshot=env.snapshot, mode="replace"),
        )

        return RepoMapWriteResult(
            success=True,
            row_count=1,
        )

    except Exception as exc:
        log.exception("Repo map write failed")
        return RepoMapWriteResult(
            success=False,
            error=str(exc),
        )


@tag(domain="ingestion", target="modules", node_type="materialize")
def t__modules(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules__scan: ModuleScanResult,
    t__modules__write_repo_map: RepoMapWriteResult,
) -> TargetRunRecord:
    """Materialize modules target with validation.

    This is the entry point for the modules target. It orchestrates
    the scan and repo_map write, then returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__modules__scan
        Scan result from upstream compute node.
    t__modules__write_repo_map
        Repo map write result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "modules")

    if executor.should_skip():
        return executor.skip()

    # Check for upstream failures
    if not t__modules__scan.success:
        return executor.fail(RuntimeError(t__modules__scan.error or "Module scan failed"))

    if not t__modules__write_repo_map.success:
        return executor.fail(
            RuntimeError(t__modules__write_repo_map.error or "Repo map write failed")
        )

    # Compute final row counts
    def compute() -> dict[str, int]:
        row_counts = dict(t__modules__scan.table_counts)
        row_counts["core.repo_map"] = t__modules__write_repo_map.row_count
        return row_counts

    return executor.execute(compute)


__all__ = [
    "ModuleScanResult",
    "RepoMapWriteResult",
    "t__modules",
    "t__modules__scan",
    "t__modules__write_repo_map",
]
