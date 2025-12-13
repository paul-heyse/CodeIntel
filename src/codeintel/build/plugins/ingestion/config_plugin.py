"""Config ingest plugin.

This module provides `ConfigIngestPlugin` that flattens
configuration files into config_values table.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import MetadataPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)


CONFIG_INGEST_METADATA = CorePluginMetadata(
    name="ingest.config",
    version="3.0.0",
    description="Flatten configuration files into config_values table.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="config",
    provides=("core.config_values",),
    requires=(),
    produces_tables=("core.config_values",),
    consumes_tables=(),
    supports_incremental=True,
    scope_aware=True,
)


class ConfigIngestPlugin(MetadataPlugin):
    """Flatten configuration files into config_values table.

    This plugin reads various configuration files (YAML, JSON, TOML, INI)
    and flattens their structure into key-value pairs.

    Outputs
    -------
    - core.config_values: Flattened config key-value pairs
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CONFIG_INGEST_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute config ingestion.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self

        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        profile = default_config_profile(ctx.repo_root)

        config_files: list[ModuleRecord] = list(
            FilesystemDiscoveryAdapter.discover_modules(ctx.repo_root, profile)
        )

        if not config_files:
            log.info("No config files found matching profile")
            return TargetResult.succeeded(row_counts={})

        step = ConfigIngestStep(storage=storage, discovery=discovery)
        result = step.execute(
            config_files,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if result.errors:
            for error in result.errors:
                log.warning("Config parse warning: %s", error)

        if result.rows_written > 0 or not result.errors:
            return TargetResult.succeeded(row_counts=result.table_counts or {})

        errors = "; ".join(result.errors)
        return TargetResult.failed(f"Config ingest failed: {errors}")


__all__ = [
    "CONFIG_INGEST_METADATA",
    "ConfigIngestPlugin",
]
