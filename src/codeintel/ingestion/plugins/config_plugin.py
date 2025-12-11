"""Config ingest plugin.

This module provides `ConfigIngestPlugin` that flattens
configuration files into config_values table.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage
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
        stage=cast("PluginStage", core.stage or "config"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


class ConfigIngestPlugin(TargetPlugin):
    """Flatten configuration files into config_values table.

    This plugin reads various configuration files (YAML, JSON, TOML, INI)
    and flattens their structure into key-value pairs.

    Outputs
    -------
    - core.config_values: Flattened config key-value pairs
    """

    plugin_name: ClassVar[str] = "config_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Flatten configuration files into config_values table."
    _core_metadata: ClassVar[CorePluginMetadata] = CONFIG_INGEST_METADATA

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
        _ = self  # Protocol method requires instance

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Create config profile for config files (YAML, JSON, TOML, INI)
        profile = default_config_profile(ctx.repo_root)

        # Find config files using config_profile
        config_files: list[ModuleRecord] = list(
            FilesystemDiscoveryAdapter.discover_modules(ctx.repo_root, profile)
        )

        if not config_files:
            log.info("No config files found matching profile")
            return TargetResult.succeeded(row_counts={})

        # Execute step
        step = ConfigIngestStep(storage=storage, discovery=discovery)
        result = step.execute(
            config_files,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        # Log parse errors as warnings but don't fail if we processed some files
        if result.errors:
            for error in result.errors:
                log.warning("Config parse warning: %s", error)

        # Consider it a success if we wrote any rows, even with some parse failures
        if result.rows_written > 0 or not result.errors:
            return TargetResult.succeeded(row_counts=result.table_counts or {})

        # Only fail if there were errors AND no data was written
        errors = "; ".join(result.errors)
        return TargetResult.failed(f"Config ingest failed: {errors}")


__all__ = [
    "CONFIG_INGEST_METADATA",
    "ConfigIngestPlugin",
]
