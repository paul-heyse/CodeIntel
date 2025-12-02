"""Repository scanning step with port injection.

This module provides a pure domain logic implementation for scanning
repository modules and building change tracker state, using ports
for all I/O operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.steps.base import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort, ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)


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
        """
        self._storage = storage
        self._discovery = discovery
        self._change_detection = change_detection

    def execute(
        self,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        profile: ScanProfile,
        full_rebuild: bool = False,
    ) -> tuple[StepResult, Sequence[ModuleRecord], ChangeSet]:
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
        tuple[StepResult, Sequence[ModuleRecord], ChangeSet]
            Execution result, discovered modules, and change set.
        """
        created_at = datetime.now(UTC)

        # Discover modules
        modules = self._discovery.discover_modules(repo_root, profile)
        log.info("Discovered %d modules in %s", len(modules), repo_root)

        # Compute changes
        change_request = ChangeRequest(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=profile,
        )
        change_set = self._change_detection.compute_changes(change_request, modules)

        # Build module rows
        module_rows: list[list[object]] = [
            [repo, commit, module.rel_path, module.module_name, "python", created_at]
            for module in modules
        ]

        # Build file state rows (for repo_map)
        file_state_rows: list[list[object]] = []
        for module in modules:
            digest = self._change_detection.compute_file_digest(module.file_path)
            if digest is not None:
                file_state_rows.append([
                    repo,
                    commit,
                    module.rel_path,
                    "python",
                    digest.size_bytes,
                    digest.mtime_ns,
                    digest.content_hash,
                ])

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if module_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch("core.modules", module_rows, scope=scope)
            table_counts["core.modules"] = result.rows_written
            total_rows += result.rows_written

        if file_state_rows:
            result = self._storage.write_batch("core.file_state", file_state_rows)
            table_counts["core.file_state"] = result.rows_written
            total_rows += result.rows_written

        log.info(
            "Repo scan: repo=%s commit=%s modules=%d added=%d modified=%d deleted=%d",
            repo,
            commit,
            len(modules),
            len(change_set.added),
            len(change_set.modified),
            len(change_set.deleted),
        )

        step_result = StepResult(
            rows_written=total_rows,
            table_counts=table_counts,
        )

        return step_result, modules, change_set


__all__ = ["RepoScanStep"]
