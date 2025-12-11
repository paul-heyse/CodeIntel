"""Coverage ingest plugin.

This module provides `CoverageIngestPlugin` that loads coverage.py
data and populates analytics.coverage_lines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage

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


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "tests"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
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


class CoverageIngestPlugin(TargetPlugin):
    """Load coverage.py data and populate analytics.coverage_lines.

    This plugin ingests test coverage data from coverage.py's database
    or JSON export.

    Outputs
    -------
    - analytics.coverage_lines: Line-level coverage data
    """

    plugin_name: ClassVar[str] = "coverage_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Load coverage.py data and populate analytics.coverage_lines."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = COVERAGE_INGEST_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata definition."""
        return self._core_metadata

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
        _ = self  # Protocol method requires instance

        # Resolve coverage file
        coverage_path = resolve_coverage_file(ctx)
        if coverage_path is None:
            log.info("No coverage file found, skipping coverage ingestion")
            return TargetResult.succeeded(row_counts={})

        # Get module paths and convert to ModuleRecord
        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(
            coverage_collector=ctx.resources.coverage_collector,
        )

        # Execute step
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
