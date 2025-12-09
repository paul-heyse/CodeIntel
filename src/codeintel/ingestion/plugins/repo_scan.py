"""Repository scan plugin.

This module provides `RepoScanPlugin` that scans repository modules
and builds change-tracker state. Part of the build system.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

import duckdb

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.tracker import ChangeTracker

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class RepoScanPlugin(TargetPlugin):
    """Scan repository modules and build change-tracker state.

    This plugin scans the repository tree, discovering Python modules
    and tracking changes for incremental processing.

    Outputs
    -------
    - core.file_state: File state and hashes
    - core.modules: Discovered Python modules
    - core.repo_map: Repository mapping
    - analytics.tags_index: Tag index for search
    """

    plugin_name: ClassVar[str] = "repo_scan"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Scan repository modules and build change-tracker state."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute repository scan.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        # Create scan profile - use default code profile for Python files
        profile = default_code_profile(ctx.repo_root)

        # Execute step
        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
        )

        _result, modules, _change_set = step.execute(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            profile=profile,
            full_rebuild=False,
        )

        # Build change request for tracker
        change_request = ChangeRequest(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            language="python",
            full_rebuild=False,
            scan_profile=profile,
        )

        # Create change tracker (stored in resources for downstream plugins)
        tracker = ChangeTracker.create(
            gateway=ctx.gateway,
            change_request=change_request,
            modules=modules,
            policy=None,
            change_detection=change_detection,
        )

        # Store tracker in context resources for downstream plugins
        # This is accessed via ctx.resources.change_tracker
        ctx.resources.change_tracker = tracker

        # Write repo_map entry for this repo/commit
        self._write_repo_map(ctx, modules)

        # Compute row counts from tables
        row_counts = self._compute_row_counts(ctx)

        return TargetResult.succeeded(row_counts=row_counts)

    @staticmethod
    def _write_repo_map(
        ctx: TargetExecutionContext,
        modules: Sequence[object],
    ) -> None:
        """Write repo_map entry for this scan.

        Parameters
        ----------
        ctx
            Execution context.
        modules
            List of discovered module records.
        """
        generated_at = datetime.now(tz=UTC).isoformat()
        # Extract module names from ModuleRecord objects
        module_names = [
            getattr(m, "name", str(m)) if hasattr(m, "name") else str(m) for m in modules
        ]
        modules_json = json.dumps(sorted(module_names))
        overlays_json = json.dumps({})

        # Delete existing entry for this repo/commit
        ctx.gateway.con.execute(
            "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        )

        # Insert new entry
        ctx.gateway.core.insert_repo_map(
            [(ctx.repo, ctx.commit, modules_json, overlays_json, generated_at)]
        )

    @staticmethod
    def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
        """Compute row counts for output tables.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, int]
            Row counts per table.
        """
        row_counts: dict[str, int] = {}
        for table_key in ctx.contract.table_keys:
            try:
                count = ctx.gateway.con.execute(
                    f"SELECT COUNT(*) FROM {table_key} "  # noqa: S608
                    f"WHERE repo = ? AND commit = ?",
                    [ctx.repo, ctx.commit],
                ).fetchone()
                row_counts[table_key] = int(count[0]) if count else 0
            except (RuntimeError, OSError, duckdb.CatalogException) as exc:
                log.warning("Row count fallback for %s: %s", table_key, exc)
                row_counts[table_key] = 0
        return row_counts


__all__ = ["RepoScanPlugin"]
