"""AST extraction plugin using class-based architecture.

This module provides `AstExtractPlugin`, a class-based plugin that
parses Python AST and persists rows + metrics into core.ast_* tables.

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
    from collections.abc import Sequence

    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)


@dataclass
class AstExtractPlugin(TrackerRequiringPlugin, TableWriterIngestPlugin, WithDependencyData):
    """Parse Python AST and persist rows + metrics.

    This plugin parses Python source files using the stdlib AST module,
    extracting node information and computing metrics. Results are
    persisted to core.ast_nodes and core.ast_metrics tables.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("ast_extract").
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

    plugin_name: ClassVar[str] = "ast_extract"
    plugin_description: ClassVar[str] = (
        "Parse Python AST and persist rows + metrics into core.ast_* tables."
    )
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = (
        "core.ast_nodes",
        "core.ast_metrics",
    )

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
        """Execute AST extraction.

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
        from codeintel.ingestion.steps import AstExtractStep

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Get modules from tracker or scratch
        modules = self._get_modules(ctx)

        # Execute step
        step = AstExtractStep(storage=storage, discovery=discovery)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if result.errors:
            for error in result.errors:
                log.warning("AST extraction error: %s", error)

        return result.table_counts

    def _get_modules(self, ctx: IngestExecutionContext) -> Sequence[ModuleRecord]:
        """Get module list from tracker or inventory.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Sequence[ModuleRecord]
            Sequence of ModuleRecord instances.
        """
        from codeintel.ingestion.common import iter_modules
        from codeintel.storage.module_index import load_module_map

        module_map = load_module_map(
            ctx.gateway,
            ctx.repo,
            ctx.commit,
            language="python",
            logger=log,
        )

        return list(
            iter_modules(
                module_map,
                ctx.repo_root,
                logger=log,
                scan_profile=ctx.code_profile,
            )
        )


__all__ = ["AstExtractPlugin"]
