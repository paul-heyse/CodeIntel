"""Docstrings ingest plugin using class-based architecture.

This module provides `DocstringsIngestPlugin`, a class-based plugin that
extracts docstrings and persists structured rows into core.docstrings.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.core.base import TableWriterIngestPlugin, TrackerRequiringPlugin
from codeintel.ingestion.core.traits import WithDependencyData
from codeintel.ingestion.plugins.protocol import (
    IngestResourceHints,
    IngestStage,
)

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.ports.discovery import ModuleRecord

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
    resource_hints : IngestResourceHints
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

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
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
        from codeintel.ingestion.adapters import (
            DuckDBStorageAdapter,
            FilesystemDiscoveryAdapter,
        )
        from codeintel.ingestion.steps import DocstringsExtractStep

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Get modules
        modules = self._get_modules(ctx)

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

    def _get_modules(self, ctx: IngestExecutionContext) -> list[ModuleRecord]:
        """Get module list from tracker or inventory.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[ModuleRecord]
            List of module records.
        """
        from codeintel.ingestion.common import iter_modules
        from codeintel.ingestion.ports.discovery import ModuleRecord
        from codeintel.storage.module_index import load_module_map

        module_map = load_module_map(
            ctx.gateway,
            ctx.repo,
            ctx.commit,
            language="python",
            logger=log,
        )

        return [
            m
            for m in iter_modules(
                module_map,
                ctx.repo_root,
                logger=log,
                scan_profile=ctx.code_profile,
            )
            if isinstance(m, ModuleRecord)
        ]


__all__ = ["DocstringsIngestPlugin"]
