"""Coverage ingest plugin.

This module provides `CoverageIngestPlugin` that loads coverage.py
data and populates analytics.coverage_lines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


COVERAGE_INGEST_METADATA = CorePluginMetadata(
    name="ingest.coverage",
    version="3.0.0",
    description="Load coverage.py data and populate analytics.coverage_lines.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="tests",
    provides=("analytics.coverage_lines",),
    requires=("core.modules",),
    produces_tables=("analytics.coverage_lines",),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


def resolve_coverage_file(ctx: TargetExecutionContext) -> Path | None:
    """Resolve the coverage data file.

    Returns
    -------
    Path | None
        Path to coverage file or None if not found.
    """
    candidates = [
        ctx.repo_root / ".coverage",
        ctx.repo_root / "coverage.json",
        ctx.build_dir / "coverage.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class CoverageIngestPlugin(MetadataPlugin):
    """Load coverage.py data and populate analytics.coverage_lines.

    This plugin ingests test coverage data from coverage.py's database
    or JSON export.

    Outputs
    -------
    - analytics.coverage_lines: Line-level coverage data
    """

    _core_metadata: ClassVar[CorePluginMetadata] = COVERAGE_INGEST_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute coverage ingestion.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self

        coverage_path = resolve_coverage_file(ctx)
        if coverage_path is None:
            log.info("No coverage file found, skipping coverage ingestion")
            return TargetResult.succeeded(row_counts={})

        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(
            coverage_collector=ctx.resources.coverage_collector,
        )

        step = CoverageIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            coverage_file=coverage_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"Coverage ingest failed: {errors}")

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = [
    "COVERAGE_INGEST_METADATA",
    "CoverageIngestPlugin",
    "get_module_paths",
    "paths_to_modules",
    "resolve_coverage_file",
]
