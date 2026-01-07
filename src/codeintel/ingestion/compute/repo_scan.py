"""Repository scanning step with port injection.

This module provides a pure domain logic implementation for scanning
repository modules and building change tracker state, using ports
for all I/O operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.columnar.rows import (
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_table_for_table,
    table_for_columnar_rows,
)
from codeintel.ingestion.context import (
    IngestionContext,
    resolve_repo_commit,
    resolve_repo_root,
    resolve_scan_profile,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import pyarrow as pa

    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort, ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

log = logging.getLogger(__name__)
MODULES_TABLE_KEY = "core.modules"
FILE_STATE_TABLE_KEY = "core.file_state"
REPO_MAP_TABLE_KEY = "core.repo_map"


@dataclass(frozen=True)
class RepoScanResult:
    """Result from repository scanning.

    Attributes
    ----------
    modules
        Discovered module records.
    change_set
        Change set describing added/modified/deleted modules.
    module_rows
        Columnar rows for core.modules.
    file_state_rows
        Columnar rows for core.file_state.
    repo_map_rows
        Columnar rows for core.repo_map.
    """

    modules: tuple[ModuleRecord, ...]
    change_set: ChangeSet
    module_rows: ColumnarRows
    file_state_rows: ColumnarRows
    repo_map_rows: ColumnarRows
    module_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(MODULES_TABLE_KEY)
    )
    file_state_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(FILE_STATE_TABLE_KEY)
    )
    repo_map_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(REPO_MAP_TABLE_KEY)
    )


class RepoScanStep:
    """Repository scanning step with port injection.

    This step scans repository modules and builds change tracker state,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for finding modules.
    change_detection
        Change detection port for computing changes.
    """

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        change_detection: ChangeDetectionPort,
        module_filter: Callable[[Sequence[ModuleRecord]], Sequence[ModuleRecord]] | None = None,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        discovery
            Discovery port for finding modules.
        change_detection
            Change detection port for computing changes.
        module_filter
            Optional filter applied to discovered modules before persistence.
        """
        self._discovery = discovery
        self._change_detection = change_detection
        self._module_filter = module_filter

    def execute(
        self,
        *,
        snapshot: SnapshotRef | None = None,
        repo_root: Path | None = None,
        profile: ScanProfile | None = None,
        full_rebuild: bool = False,
        context: IngestionContext | None = None,
    ) -> RepoScanResult:
        """Execute repository scanning.

        Parameters
        ----------
        snapshot
            Optional snapshot reference (provides repo/commit/root when set).
        repo_root
            Repository root path.
        profile
            Scan profile for module discovery.
        full_rebuild
            Whether to force a full rebuild.
        context
            Optional ingestion context for repo/commit/root resolution.

        Returns
        -------
        RepoScanResult
            Discovered modules, change set, and row tuples.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=snapshot.repo if snapshot is not None else None,
            commit=snapshot.commit if snapshot is not None else None,
        )
        resolved_root = resolve_repo_root(
            context=context,
            repo_root=snapshot.repo_root if snapshot is not None else repo_root,
        )
        resolved_profile = resolve_scan_profile(context=context, scan_profile=profile)

        modules = list(self._discovery.discover_modules(resolved_root, resolved_profile))
        if self._module_filter is not None:
            modules = list(self._module_filter(modules))
        modules = _dedupe_modules(modules)
        log.info("Discovered %d modules in %s", len(modules), resolved_root)

        change_request = ChangeRequest(
            repo=resolved_repo,
            commit=resolved_commit,
            repo_root=resolved_root,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=resolved_profile,
        )
        change_set = self._change_detection.compute_changes(change_request, modules)

        module_buffer = columnar_buffer_for_table_key(MODULES_TABLE_KEY)
        for module in modules:
            payload = {
                "module": module.module_name,
                "path": module.rel_path,
                "repo": resolved_repo,
                "commit": resolved_commit,
                "language": "python",
                "tags": [],
                "owners": [],
                "row_hash": None,
            }
            module_buffer.append(payload)

        repo_map_rows = self._build_repo_map_rows(
            repo=resolved_repo,
            commit=resolved_commit,
            modules=modules,
        )
        module_rows_reader, _ = table_for_columnar_rows(
            MODULES_TABLE_KEY,
            module_buffer.data,
            extras_policy="retain",
        )
        file_state_rows_reader, _ = table_for_columnar_rows(
            FILE_STATE_TABLE_KEY,
            change_set.state_rows,
            extras_policy="retain",
        )
        repo_map_rows_reader, _ = table_for_columnar_rows(
            REPO_MAP_TABLE_KEY,
            repo_map_rows,
            extras_policy="retain",
        )

        log.info(
            "Repo scan: repo=%s commit=%s modules=%d added=%d modified=%d deleted=%d",
            resolved_repo,
            resolved_commit,
            len(modules),
            len(change_set.added),
            len(change_set.modified),
            len(change_set.deleted),
        )

        return RepoScanResult(
            modules=tuple(modules),
            change_set=change_set,
            module_rows=module_buffer.data,
            file_state_rows=change_set.state_rows,
            repo_map_rows=repo_map_rows,
            module_rows_reader=module_rows_reader,
            file_state_rows_reader=file_state_rows_reader,
            repo_map_rows_reader=repo_map_rows_reader,
        )

    @staticmethod
    def _build_repo_map_rows(
        *,
        repo: str,
        commit: str,
        modules: Sequence[ModuleRecord],
    ) -> ColumnarRows:
        if not modules:
            return {}
        module_entries: dict[str, str] = {}
        for module in modules:
            module_entries[str(module.module_name)] = str(module.rel_path)
        buffer = columnar_buffer_for_table_key(REPO_MAP_TABLE_KEY)
        buffer.append(
            {
                "repo": repo,
                "commit": commit,
                "modules": module_entries,
                "overlays": {},
                "generated_at": datetime.now(tz=UTC),
            }
        )
        return buffer.data


__all__ = ["RepoScanResult", "RepoScanStep"]


def _dedupe_modules(modules: Sequence[ModuleRecord]) -> list[ModuleRecord]:
    deduped: dict[tuple[str, str], ModuleRecord] = {}
    for module in modules:
        key = (module.module_name, module.rel_path)
        if key not in deduped:
            deduped[key] = module
    return list(deduped.values())
