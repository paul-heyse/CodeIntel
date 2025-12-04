"""Coverage ingest plugin using class-based architecture.

This module provides `CoverageIngestPlugin`, a class-based plugin that
loads coverage.py data and populates analytics.coverage_lines.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.adapters import DuckDBStorageAdapter, ToolRunnerAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.core.base import (
    TableWriterIngestPlugin,
    ToolDependentIngestPlugin,
    TrackerRequiringPlugin,
)
from codeintel.ingestion.core.traits import WithDependencyData, WithToolDependencies
from codeintel.ingestion.plugins.protocol import (
    IngestPluginResult,
    IngestResourceHints,
    IngestStage,
)
from codeintel.ingestion.resources import ModuleProvider, ToolsProvider

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

log = logging.getLogger(__name__)


@dataclass
class CoverageIngestPlugin(
    TrackerRequiringPlugin,
    ToolDependentIngestPlugin,
    TableWriterIngestPlugin,
    WithDependencyData,
    WithToolDependencies,
):
    """Load coverage.py data and populate analytics.coverage_lines.

    This plugin ingests test coverage data from coverage.py's database
    or JSON export.

    Class Attributes
    ----------------
    plugin_name : str
        Stable identifier ("coverage_ingest").
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

    plugin_name: ClassVar[str] = "coverage_ingest"
    plugin_description: ClassVar[str] = (
        "Load coverage.py data and populate analytics.coverage_lines."
    )
    plugin_stage: ClassVar[IngestStage] = "enrich"
    plugin_version: ClassVar[str] = "2.0.0"

    output_tables: ClassVar[tuple[str, ...]] = ("analytics.coverage_lines",)

    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    tool_dependencies: ClassVar[tuple[str, ...]] = ("coverage",)
    supports_incremental: ClassVar[bool] = True
    tracker_required: ClassVar[bool] = False
    tool_required: ClassVar[bool] = False

    resource_hints: ClassVar[IngestResourceHints] = IngestResourceHints(
        cpu_intensive=False,
        io_intensive=True,
    )

    def compute(
        self,
        ctx: IngestExecutionContext,
    ) -> Mapping[str, int] | None:
        """Execute coverage ingestion.

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
        _SkipError
            When coverage file is missing (handled by execute).
        RuntimeError
            When coverage ingestion fails.
        """
        # Resolve coverage file
        coverage_path = self._resolve_coverage_file(ctx)
        if coverage_path is None:
            msg = "missing_coverage_file"
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

        # Execute step (async)
        step = CoverageIngestStep(storage=storage, tools=tool)
        result = asyncio.run(
            step.execute_async(
                modules,
                repo=ctx.repo,
                commit=ctx.commit,
                repo_root=ctx.repo_root,
                coverage_file=coverage_path,
            )
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            msg = f"Coverage ingest failed: {errors}"
            raise RuntimeError(msg)

        return result.table_counts

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

    @staticmethod
    def _resolve_coverage_file(ctx: IngestExecutionContext) -> Path | None:
        """Resolve the coverage data file.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        Path | None
            Path to coverage file or None if not found.
        """
        # Check common locations
        candidates = [
            ctx.repo_root / ".coverage",
            ctx.repo_root / "coverage.json",
            ctx.paths.coverage_json if hasattr(ctx.paths, "coverage_json") else None,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        return None


class _SkipError(Exception):
    """Internal signal to indicate plugin should skip."""


__all__ = ["CoverageIngestPlugin"]
