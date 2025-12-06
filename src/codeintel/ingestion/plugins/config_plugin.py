"""Config ingest plugin.

This module provides `ConfigIngestPlugin` that flattens
configuration files into config_values table.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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


__all__ = ["ConfigIngestPlugin"]
