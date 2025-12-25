"""Repository scanning step with port injection.

This module provides a pure domain logic implementation for scanning
repository modules and building change tracker state, using ports
for all I/O operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.hashing import stable_hash
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.ports.change_detection import ChangeRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort, ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

log = logging.getLogger(__name__)
MODULES_TABLE_KEY = "core.modules"
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
        Row tuples for core.modules.
    file_state_rows
        Row tuples for core.file_state.
    repo_map_rows
        Row tuples for core.repo_map.
    """

    modules: tuple[ModuleRecord, ...]
    change_set: ChangeSet
    module_rows: tuple[tuple[object, ...], ...]
    file_state_rows: tuple[tuple[object, ...], ...]
    repo_map_rows: tuple[tuple[object, ...], ...]


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
        repo: str,
        commit: str,
        repo_root: Path,
        profile: ScanProfile,
        full_rebuild: bool = False,
    ) -> RepoScanResult:
        """Execute repository scanning.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.
        profile
            Scan profile for module discovery.
        full_rebuild
            Whether to force a full rebuild.

        Returns
        -------
        RepoScanResult
            Discovered modules, change set, and row tuples.
        """
        modules = list(self._discovery.discover_modules(repo_root, profile))
        if self._module_filter is not None:
            modules = list(self._module_filter(modules))
        modules = _dedupe_modules(modules)
        log.info("Discovered %d modules in %s", len(modules), repo_root)

        change_request = ChangeRequest(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=profile,
        )
        change_set = self._change_detection.compute_changes(change_request, modules)

        serializer = row_serializer_for_table_key(MODULES_TABLE_KEY)
        module_rows: list[tuple[object, ...]] = []
        for module in modules:
            payload = {
                "module": module.module_name,
                "path": module.rel_path,
                "repo": repo,
                "commit": commit,
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            }
            payload["row_hash"] = stable_hash(payload)
            module_rows.append(serializer(payload))

        repo_map_rows = self._build_repo_map_rows(
            repo=repo,
            commit=commit,
            modules=modules,
        )

        log.info(
            "Repo scan: repo=%s commit=%s modules=%d added=%d modified=%d deleted=%d",
            repo,
            commit,
            len(modules),
            len(change_set.added),
            len(change_set.modified),
            len(change_set.deleted),
        )

        return RepoScanResult(
            modules=tuple(modules),
            change_set=change_set,
            module_rows=tuple(module_rows),
            file_state_rows=tuple(change_set.state_rows),
            repo_map_rows=repo_map_rows,
        )

    @staticmethod
    def _build_repo_map_rows(
        *,
        repo: str,
        commit: str,
        modules: Sequence[ModuleRecord],
    ) -> tuple[tuple[object, ...], ...]:
        if not modules:
            return ()
        module_entries: dict[str, str] = {}
        for module in modules:
            module_entries[str(module.module_name)] = str(module.rel_path)
        serializer = row_serializer_for_table_key(REPO_MAP_TABLE_KEY)
        return (
            serializer(
                {
                    "repo": repo,
                    "commit": commit,
                    "modules": json.dumps(module_entries),
                    "overlays": json.dumps({}),
                    "generated_at": datetime.now(tz=UTC),
                }
            ),
        )


__all__ = ["RepoScanResult", "RepoScanStep"]


def _dedupe_modules(modules: Sequence[ModuleRecord]) -> list[ModuleRecord]:
    deduped: dict[tuple[str, str], ModuleRecord] = {}
    for module in modules:
        key = (module.module_name, module.rel_path)
        if key not in deduped:
            deduped[key] = module
    return list(deduped.values())
