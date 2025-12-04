"""Repository scan plugin using class-based architecture.

This module provides `RepoScanPlugin`, a class-based plugin that scans
repository modules and builds change-tracker state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.traits import WithDependencyData, WithRowCounts
from codeintel.ingestion.plugins.protocol import (
    IngestPluginResult,
    IngestResourceHints,
    IngestStage,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.tracker import ChangeTracker

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class RepoScanPlugin(BaseIngestPlugin, WithDependencyData, WithRowCounts):
    """Scan repository modules and build change-tracker state.

    This plugin scans the repository tree, discovering Python modules
    and tracking changes for incremental processing. Results are stored
    in scratch for downstream plugins.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("repo_scan").
    plugin_description : str
        Human-readable description.
    plugin_stage : IngestStage
        Processing stage ("scan").
    output_tables : tuple[str, ...]
        Tables written to.
    provides : tuple[str, ...]
        Capabilities provided.
    supports_incremental : bool
        Whether incremental mode is supported.
    resource_hints : IngestResourceHints
        Resource requirements.
    """

    plugin_name: ClassVar[str] = "repo_scan"
    plugin_description: ClassVar[str] = "Scan repository modules and build change-tracker state."
    plugin_stage: ClassVar[IngestStage] = "scan"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = (
        "core.file_state",
        "core.modules",
        "core.repo_map",
        "analytics.tags_index",
    )

    provides: ClassVar[tuple[str, ...]] = ("modules", "change_tracker")
    supports_incremental: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=False,
        io_intensive=True,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute repository scan.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Row counts after scan, or None for auto-compute.
        """
        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.snapshot.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        # Execute step
        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
        )

        _result, modules, _change_set = step.execute(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            repo_root=ctx.snapshot.repo_root,
            profile=ctx.validated_code_profile,
            full_rebuild=False,
        )

        # Build change request for tracker
        change_request = ChangeRequest(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            repo_root=ctx.snapshot.repo_root,
            language="python",
            full_rebuild=False,
            scan_profile=ctx.validated_code_profile,
        )

        # Create change tracker
        tracker = ChangeTracker.create(
            gateway=ctx.gateway,
            change_request=change_request,
            modules=modules,
            policy=None,
            change_detection=change_detection,
        )

        # Store tracker in scratch for downstream plugins
        self.set_dependency_data(ctx, "change_tracker", tracker)

        # Return None to trigger auto row count computation
        return None

    def _build_success_result(
        self,
        row_counts: Mapping[str, int] | None,
        ctx: IngestExecutionContext,
    ) -> IngestPluginResult:
        """Build success result with auto row counts.

        Parameters
        ----------
        row_counts
            Explicit row counts or None.
        ctx
            Execution context.

        Returns
        -------
        IngestPluginResult
            Success result.
        """
        if row_counts is None:
            row_counts = self.compute_row_counts_for_tables(ctx)
        return IngestPluginResult.ok(row_counts=dict(row_counts))


__all__ = ["RepoScanPlugin"]
