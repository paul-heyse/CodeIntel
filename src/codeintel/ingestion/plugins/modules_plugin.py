"""Module ingest plugin."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, SupportsInt, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
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
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage

log = logging.getLogger(__name__)


MODULE_INGEST_METADATA = CorePluginMetadata(
    name="ingest.modules",
    version="2.0.0",
    description="Discover and ingest Python modules.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="goid",
    provides=("core.modules",),
    requires=(),
    produces_tables=("core.modules",),
    consumes_tables=(),
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
        stage=cast("PluginStage", core.stage or "goid"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


class ModuleIngestPlugin(TargetPlugin):
    """Discover and ingest Python modules."""

    plugin_name: ClassVar[str] = "modules"
    plugin_version: ClassVar[str] = "2.0.0"
    plugin_description: ClassVar[str] = "Discover and ingest Python modules."
    _core_metadata: ClassVar[CorePluginMetadata] = MODULE_INGEST_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata.

        Returns
        -------
        PluginMetadata
            Protocol metadata facade.
        """
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return core metadata.

        Returns
        -------
        CorePluginMetadata
            Canonical metadata definition.
        """
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
        """Execute module ingestion via repository scan.

        Returns
        -------
        TargetResult
            Execution result with row counts.
        """
        _ = self  # Protocol method requires instance
        opts = self.resolve_options()

        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        profile = build_scan_profile(ctx.repo_root, opts)

        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
            module_filter=lambda discovered: filter_modules(discovered, opts),
        )

        result, modules, _change_set = step.execute(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            profile=profile,
            full_rebuild=False,
        )

        if not result.success:
            return TargetResult.failed("Module ingest failed during repo scan")

        change_request = ChangeRequest(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            language="python",
            full_rebuild=False,
            scan_profile=profile,
        )

        tracker = ChangeTracker.create(
            gateway=ctx.gateway,
            change_request=change_request,
            modules=modules,
            policy=None,
            change_detection=change_detection,
        )
        ctx.resources.change_tracker = tracker

        row_counts = self._compute_row_counts(ctx)

        return TargetResult.succeeded(row_counts=row_counts)

    @staticmethod
    def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
        """Compute row counts for output tables.

        Returns
        -------
        dict[str, int]
            Row counts keyed by table name.
        """
        row_counts: dict[str, int] = {}
        for table_key in ctx.contract.table_keys:
            try:
                table = ctx.gateway.ibis.table(table_key)
                count_expr = filter_by(
                    table,
                    ibis_bool(table.repo == ctx.repo),
                    ibis_bool(table.commit == ctx.commit),
                ).count()
                count = count_expr.execute()
                row_counts[table_key] = int(cast("SupportsInt", count))
            except (RuntimeError, OSError, DuckDBCatalogException) as exc:
                log.warning("Row count fallback for %s: %s", table_key, exc)
                row_counts[table_key] = 0
        return row_counts

    @staticmethod
    def _write_repo_map(
        ctx: TargetExecutionContext,
        modules: list[object],
    ) -> None:
        """Write repo_map entry for this scan."""
        generated_at = datetime.now(tz=UTC).isoformat()
        module_names = [
            getattr(module, "name", str(module)) if hasattr(module, "name") else str(module)
            for module in modules
        ]
        modules_json = json.dumps(sorted(module_names))
        overlays_json = json.dumps({})

        backend = DuckDBPolicyBackend(ctx.gateway)
        backend.delete_for_snapshot("core.repo_map", repo=ctx.repo, commit=ctx.commit)
        ctx.gateway.core.insert_repo_map(
            [(ctx.repo, ctx.commit, modules_json, overlays_json, generated_at)]
        )


__all__ = [
    "MODULE_INGEST_METADATA",
    "ModuleIngestPlugin",
]
