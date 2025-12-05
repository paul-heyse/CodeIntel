"""Docstrings ingest plugin using class-based architecture.

This module provides `DocstringsIngestPlugin`, a class-based plugin that
extracts docstrings and persists structured rows into core.docstrings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.core.plugins.types.protocol import PluginResourceHints
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute import DocstringsExtractStep
from codeintel.ingestion.core.base import TableWriterIngestPlugin, TrackerRequiringPlugin
from codeintel.ingestion.core.traits import WithDependencyData
from codeintel.ingestion.plugins.protocol import (
    IngestStage,
)
from codeintel.ingestion.resources import ModuleProvider

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class DocstringsIngestPlugin(
    TrackerRequiringPlugin,
    TableWriterIngestPlugin,
    WithDependencyData,
):
    """Extract docstrings and persist structured rows into core.docstrings.

    This plugin parses Python source files to extract docstrings from
    modules, classes, and functions, persisting structured information
    for documentation analysis.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("docstrings_ingest").
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
    resource_hints : PluginResourceHints
        Resource requirements.
    """

    plugin_name: ClassVar[str] = "docstrings_ingest"
    plugin_description: ClassVar[str] = (
        "Extract docstrings and persist structured rows into core.docstrings."
    )
    plugin_stage: ClassVar[IngestStage] = "enrich"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = ("core.docstrings",)

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    supports_incremental: ClassVar[bool] = True
    tracker_required: ClassVar[bool] = False

    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
        cpu_intensive=True,
        io_intensive=False,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute docstring extraction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Row counts per table.
        """
        _ = self  # Required by interface, accessed via ctx

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Get modules from provider
        modules_provider = ctx.require(ModuleProvider)
        modules = list(modules_provider.get())

        # Execute step
        step = DocstringsExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if result.errors:
            for error in result.errors:
                log.warning("Docstring extraction error: %s", error)

        return result.table_counts


__all__ = ["DocstringsIngestPlugin"]
