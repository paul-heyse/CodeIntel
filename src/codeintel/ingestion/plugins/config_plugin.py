"""Config ingest plugin using class-based architecture.

This module provides `ConfigIngestPlugin`, a class-based plugin that
flattens configuration files into config_values table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.core.base import TableWriterIngestPlugin, TrackerRequiringPlugin
from codeintel.ingestion.core.traits import WithDependencyData
from codeintel.ingestion.plugins.protocol import (
    IngestResourceHints,
    IngestStage,
)
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class ConfigIngestPlugin(
    TrackerRequiringPlugin,
    TableWriterIngestPlugin,
    WithDependencyData,
):
    """Flatten configuration files into config_values table.

    This plugin reads various configuration files (YAML, JSON, TOML, INI)
    and flattens their structure into key-value pairs.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("config_ingest").
    plugin_description : str
        Human-readable description.
    plugin_stage : IngestStage
        Processing stage ("enrich").
    output_tables : tuple[str, ...]
        Tables written to.
    depends_on : tuple[str, ...]
        Plugin dependencies.
    requires : tuple[str, ...]
        Required capabilities.
    supports_incremental : bool
        Whether incremental mode is supported.
    resource_hints : IngestResourceHints
        Resource requirements.
    """

    plugin_name: ClassVar[str] = "config_ingest"
    plugin_description: ClassVar[str] = "Flatten configuration files into config_values table."
    plugin_stage: ClassVar[IngestStage] = "enrich"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = ("core.config_values",)

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    supports_incremental: ClassVar[bool] = True
    tracker_required: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=False,
        io_intensive=True,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute config ingestion.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Row counts, or None for auto-compute.

        Raises
        ------
        RuntimeError
            When config ingestion fails.
        """
        _ = self  # Required by interface, accessed via ctx

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Find config files using config_profile
        config_files: list[ModuleRecord] = list(
            FilesystemDiscoveryAdapter.discover_modules(ctx.repo_root, ctx.validated_config_profile)
        )

        if not config_files:
            log.info("No config files found matching profile")
            return {}

        # Execute step
        step = ConfigIngestStep(storage=storage, discovery=discovery)
        result = step.execute(
            config_files,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            msg = f"Config ingest failed: {errors}"
            raise RuntimeError(msg)

        return result.table_counts


__all__ = ["ConfigIngestPlugin"]
