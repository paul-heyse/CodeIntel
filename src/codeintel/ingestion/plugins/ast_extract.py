"""AST extraction plugin using class-based architecture.

This module provides `AstExtractPlugin`, a class-based plugin that
parses Python AST and persists rows + metrics into core.ast_* tables.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.core.base import TableWriterIngestPlugin, TrackerRequiringPlugin
from codeintel.ingestion.core.traits import WithDependencyData
from codeintel.ingestion.plugins.protocol import (
    IngestIsolationKind,
    IngestResourceHints,
    IngestStage,
)
from codeintel.ingestion.resources import ModuleProvider
from codeintel.ingestion.steps import AstExtractStep

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

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
        _ = self  # Required by interface, accessed via ctx

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)

        # Get modules from provider
        modules_provider = ctx.require(ModuleProvider)
        modules = list(modules_provider.get())

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


__all__ = ["AstExtractPlugin"]
