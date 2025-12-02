"""Typing ingest plugin using class-based architecture.

This module provides `TypingIngestPlugin`, a class-based plugin that
populates analytics.typedness and analytics.static_diagnostics.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.core.base import (
    TableWriterIngestPlugin,
    ToolDependentIngestPlugin,
    TrackerRequiringPlugin,
)
from codeintel.ingestion.core.traits import WithDependencyData, WithToolDependencies
from codeintel.ingestion.plugins.protocol import (
    IngestResourceHints,
    IngestStage,
)

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class TypingIngestPlugin(
    TrackerRequiringPlugin,
    ToolDependentIngestPlugin,
    TableWriterIngestPlugin,
    WithDependencyData,
    WithToolDependencies,
):
    """Populate analytics.typedness and analytics.static_diagnostics.

    This plugin runs type checkers (pyright, pyrefly) and linters (ruff)
    to compute typedness scores and capture static diagnostics.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("typing_ingest").
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
    tool_dependencies : tuple[str, ...]
        External tools required.
    supports_incremental : bool
        Whether incremental mode is supported.
    resource_hints : IngestResourceHints
        Resource requirements.
    """

    plugin_name: ClassVar[str] = "typing_ingest"
    plugin_description: ClassVar[str] = (
        "Populate analytics.typedness and analytics.static_diagnostics."
    )
    plugin_stage: ClassVar[IngestStage] = "enrich"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = (
        "analytics.typedness",
        "analytics.static_diagnostics",
    )

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    tool_dependencies: ClassVar[tuple[str, ...]] = ("pyright", "pyrefly", "ruff")
    supports_incremental: ClassVar[bool] = True
    tracker_required: ClassVar[bool] = True
    tool_required: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=False,
        io_intensive=True,
        max_runtime_ms=180000,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute typing analysis.

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
            When typing analysis fails.
        """
        _ = self  # Required by interface, accessed via ctx
        import asyncio

        from codeintel.ingestion.adapters import (
            DuckDBStorageAdapter,
            FilesystemDiscoveryAdapter,
            ToolRunnerAdapter,
        )
        from codeintel.ingestion.resources import ModuleProvider, ToolsProvider
        from codeintel.ingestion.steps.typing_ingest import TypingIngestStep

        # Get tool service from provider
        tools_provider = ctx.require(ToolsProvider)
        service = tools_provider.get()

        # Get modules from provider
        modules_provider = ctx.require(ModuleProvider)
        modules = list(modules_provider.get())

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
        tools = ToolRunnerAdapter(service)

        # Execute step (async)
        step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)
        result = asyncio.run(
            step.execute_async(
                modules,
                repo=ctx.repo,
                commit=ctx.commit,
                repo_root=str(ctx.repo_root),
            )
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            msg = f"Typing ingest failed: {errors}"
            raise RuntimeError(msg)

        return result.table_counts


__all__ = ["TypingIngestPlugin"]
