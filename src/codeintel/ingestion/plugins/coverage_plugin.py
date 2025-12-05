"""Coverage ingest plugin.

This module provides `CoverageIngestPlugin` that loads coverage.py
data and populates analytics.coverage_lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


def _paths_to_modules(paths: list[str], repo_root: Path) -> list[ModuleRecord]:
    """Convert string paths to ModuleRecord objects.

    Returns
    -------
    list[ModuleRecord]
        Module records with metadata.
    """
    total = len(paths)
    return [
        ModuleRecord(
            rel_path=path,
            module_name=path.replace("/", ".").removesuffix(".py"),
            file_path=repo_root / path,
            index=i + 1,
            total=total,
        )
        for i, path in enumerate(paths)
    ]


def _get_module_paths(ctx: TargetExecutionContext) -> list[str]:
    """Get module paths from context resources or database.

    Returns
    -------
    list[str]
        List of relative module paths.
    """
    if ctx.resources.modules:
        return list(ctx.resources.modules)
    try:
        rows = ctx.gateway.con.execute(
            "SELECT rel_path FROM core.modules WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except (RuntimeError, OSError):
        return []


def _resolve_coverage_file(ctx: TargetExecutionContext) -> Path | None:
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
        coverage_path = _resolve_coverage_file(ctx)
        if coverage_path is None:
            log.info("No coverage file found, skipping coverage ingestion")
            return TargetResult.succeeded(row_counts={})

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

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


__all__ = ["CoverageIngestPlugin"]
