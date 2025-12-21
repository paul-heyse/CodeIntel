"""Repository scanning step with port injection.

This module provides a pure domain logic implementation for scanning
repository modules and building change tracker state, using ports
for all I/O operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.compute.base import ExecutionResult
from codeintel.ingestion.ports.change_detection import ChangeRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort, ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)
MODULES_TABLE_KEY = "core.modules"


class RepoScanStep:
    """Repository scanning step with port injection.

    This step scans repository modules and builds change tracker state,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for finding modules.
    change_detection
        Change detection port for computing changes.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
        change_detection: ChangeDetectionPort,
        module_filter: Callable[[Sequence[ModuleRecord]], Sequence[ModuleRecord]] | None = None,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        discovery
            Discovery port for finding modules.
        change_detection
            Change detection port for computing changes.
        module_filter
            Optional filter applied to discovered modules before persistence.
        """
        self._storage = storage
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
    ) -> tuple[ExecutionResult, Sequence[ModuleRecord], ChangeSet]:
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
        tuple[ExecutionResult, Sequence[ModuleRecord], ChangeSet]
            Execution result, discovered modules, and change set.
        """
        modules = self._discovery.discover_modules(repo_root, profile)
        if self._module_filter is not None:
            modules = list(self._module_filter(modules))
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
        module_rows: list[tuple[object, ...]] = [
            serializer(
                {
                    "module": module.module_name,
                    "path": module.rel_path,
                    "repo": repo,
                    "commit": commit,
                    "language": "python",
                    "tags": "[]",
                    "owners": "[]",
                }
            )
            for module in modules
        ]

        table_counts: dict[str, int] = {}
        if module_rows:
            scope = f"{repo}@{commit}"
            self._storage.delete_by_params(MODULES_TABLE_KEY, [repo, commit])
            result = self._storage.write_batch(MODULES_TABLE_KEY, module_rows, scope=scope)
            table_counts[MODULES_TABLE_KEY] = result.rows_affected

        log.info(
            "Repo scan: repo=%s commit=%s modules=%d added=%d modified=%d deleted=%d",
            repo,
            commit,
            len(modules),
            len(change_set.added),
            len(change_set.modified),
            len(change_set.deleted),
        )

        step_result = ExecutionResult.ok(table_counts=table_counts)

        return step_result, modules, change_set


__all__ = ["RepoScanStep"]
