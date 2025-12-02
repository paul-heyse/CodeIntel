"""CST extraction plugin using class-based architecture.

This module provides `CstExtractPlugin`, a class-based plugin that
parses CST via LibCST and writes rows into core.cst_nodes.

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
    IngestIsolationKind,
    IngestResourceHints,
    IngestStage,
)

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)


@dataclass
class CstExtractPlugin(TrackerRequiringPlugin, TableWriterIngestPlugin, WithDependencyData):
    """Parse CST via LibCST and write rows into core.cst_nodes.

    This plugin parses Python source files using LibCST, extracting
    concrete syntax tree nodes for detailed analysis.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("cst_extract").
    plugin_description : str
        Human-readable description.
    plugin_stage : IngestStage
        Processing stage ("parse").
    output_tables : tuple[str, ...]
        Tables written to.
    depends_on : tuple[str, ...]
        Plugin dependencies.
    requires : tuple[str, ...]
        Required capabilities.
    supports_incremental : bool
        Whether incremental mode is supported.
    isolation_kind : IngestIsolationKind
        Isolation requirement.
    resource_hints : IngestResourceHints
        Resource requirements.
    """

    plugin_name: ClassVar[str] = "cst_extract"
    plugin_description: ClassVar[str] = "Parse CST via LibCST and write rows into core.cst_nodes."
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = ("core.cst_nodes",)

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    supports_incremental: ClassVar[bool] = True

    isolation_kind: ClassVar[IngestIsolationKind] = "process"
    tracker_required: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=True,
        io_intensive=False,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute CST extraction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Row counts per table, or None for auto-compute.
        """
        from codeintel.ingestion.adapters import (
            DuckDBStorageAdapter,
            FilesystemDiscoveryAdapter,
        )
        from codeintel.ingestion.steps import CstExtractStep

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Get modules
        modules = self._get_modules(ctx)

        # Execute step
        step = CstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if result.errors:
            for error in result.errors:
                log.warning("CST extraction error: %s", error)

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


__all__ = ["CstExtractPlugin"]
