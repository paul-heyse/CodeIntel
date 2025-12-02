"""SCIP ingest plugin using class-based architecture.

This module provides `ScipIngestPlugin`, a class-based plugin that
runs scip-python and persists symbols and GOID crosswalk.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.core.base import (
    ToolDependentIngestPlugin,
    TrackerRequiringPlugin,
)
from codeintel.ingestion.core.traits import WithDependencyData, WithToolDependencies
from codeintel.ingestion.plugins.protocol import (
    IngestPluginResult,
    IngestResourceHints,
    IngestStage,
)

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class ScipIngestPlugin(
    TrackerRequiringPlugin,
    ToolDependentIngestPlugin,
    WithDependencyData,
    WithToolDependencies,
):
    """Run scip-python and persist symbols and GOID crosswalk.

    This plugin executes the SCIP-Python indexer to generate semantic
    code intelligence data, including symbol information and global
    identifier crosswalk.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("scip_ingest").
    plugin_description : str
        Human-readable description.
    plugin_stage : IngestStage
        Processing stage ("index").
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

    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_description: ClassVar[str] = "Run scip-python and persist symbols and GOID crosswalk."
    plugin_stage: ClassVar[IngestStage] = "index"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = (
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    )

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    tool_dependencies: ClassVar[tuple[str, ...]] = ("scip",)
    supports_incremental: ClassVar[bool] = True
    tracker_required: ClassVar[bool] = False
    tool_required: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=True,
        io_intensive=True,
        max_runtime_ms=300000,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute SCIP indexing.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Mapping[str, int] | None
            Row counts, or None on skip.

        Raises
        ------
        _SkipError
            When SCIP tools are unavailable.
        RuntimeError
            On SCIP execution failure.
        """
        _ = self  # Required by interface, accessed via ctx
        import asyncio

        from codeintel.ingestion.adapters import DuckDBStorageAdapter, ToolRunnerAdapter
        from codeintel.ingestion.resources import ModuleProvider, ToolsProvider
        from codeintel.ingestion.steps.scip_ingest import ScipIngestConfig, ScipIngestStep

        # Check if SCIP binaries are configured
        if ctx.tools.scip_python_bin is None or ctx.tools.scip_bin is None:
            msg = "SCIP binaries not configured"
            raise _SkipError(msg)

        # Get tool service from provider
        tools_provider = ctx.require(ToolsProvider)
        service = tools_provider.get()

        # Get modules from provider
        modules_provider = ctx.require(ModuleProvider)
        modules = list(modules_provider.get())

        # Create adapters
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = ToolRunnerAdapter(service)

        # Create config
        scip_dir = ctx.paths.scip_dir
        config = ScipIngestConfig(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            repo_root=ctx.snapshot.repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.json",
        )

        # Execute step (async)
        step = ScipIngestStep(storage=storage, tools=tool)
        result = asyncio.run(step.execute_async(modules, config))

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            msg = f"SCIP ingest failed: {errors}"
            raise RuntimeError(msg)

        # Return None to trigger auto row count
        return None

    def execute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
        """Execute with skip handling.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        IngestPluginResult
            Execution result.
        """
        try:
            result = self.compute(ctx)
            return self._build_success_result(result, ctx)
        except _SkipError as skip:
            return IngestPluginResult.skip(str(skip))
        except (RuntimeError, ValueError, OSError, TypeError, AttributeError) as exc:
            log.exception("Plugin %s failed", self.metadata.name)
            return IngestPluginResult.fail(f"{self.metadata.name} failed: {exc}")


class _SkipError(Exception):
    """Internal signal to indicate plugin should skip."""


__all__ = ["ScipIngestPlugin"]
