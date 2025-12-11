"""Repository scan plugin.

This module provides `RepoScanPlugin` that scans repository modules
and builds change-tracker state. Part of the build system.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginKind, PluginMetadata, PluginStage
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.plugins.helpers import build_scan_profile, filter_modules
from codeintel.ingestion.plugins.modules_options import ModuleIngestOptions
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.tracker import ChangeTracker
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway.protocol import DuckDBCatalogException

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


REPO_SCAN_METADATA = CorePluginMetadata(
    name="ingest.repo_scan",
    version="3.0.0",
    description="Scan repository modules and build change-tracker state.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="discovery",
    provides=("core.modules", "core.repo_map", "core.file_state"),
    requires=(),
    produces_tables=("core.modules", "core.repo_map", "core.file_state"),
    consumes_tables=(),
    supports_incremental=True,
    scope_aware=True,
    options_model=ModuleIngestOptions,
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "discovery"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


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
    _core_metadata: ClassVar[CorePluginMetadata] = REPO_SCAN_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata definition."""
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> ModuleIngestOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        ModuleIngestOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return ModuleIngestOptions(**dynamic_overrides)
            return ModuleIngestOptions()
        return self._options_resolver.get_options(
            self._core_metadata,
            ModuleIngestOptions,
            dynamic_overrides=dynamic_overrides,
        )

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

        opts = self.resolve_options()
        profile = build_scan_profile(ctx.repo_root, opts)

        # Execute step
        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
            module_filter=lambda discovered: filter_modules(discovered, opts),
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
        module_entries: dict[str, str] = {}
        for module in modules:
            name = getattr(module, "module_name", None) or getattr(module, "name", None)
            rel_path = getattr(module, "rel_path", None) or getattr(module, "path", None)
            if name is None:
                name = str(module)
            module_entries[str(name)] = str(rel_path) if rel_path is not None else ""
        modules_json = json.dumps(module_entries)
        overlays_json = json.dumps({})

        # Delete existing entry for this repo/commit
        policy_backend = DuckDBPolicyBackend(ctx.gateway)
        policy_backend.delete_for_snapshot("core.repo_map", repo=ctx.repo, commit=ctx.commit)
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
                table = ctx.gateway.ibis.table(table_key)
                count = (
                    table.filter((table.repo == ctx.repo) & (table.commit == ctx.commit))
                    .count()
                    .execute()
                )
                row_counts[table_key] = int(count)
            except (RuntimeError, OSError, DuckDBCatalogException) as exc:
                log.warning("Row count fallback for %s: %s", table_key, exc)
                row_counts[table_key] = 0
        return row_counts


__all__ = [
    "REPO_SCAN_METADATA",
    "RepoScanPlugin",
]
