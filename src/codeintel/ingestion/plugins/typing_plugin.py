"""Typing ingest plugin.

This module provides `TypingIngestPlugin` that populates
analytics.typedness and analytics.static_diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
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
            "SELECT path FROM core.modules WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except (RuntimeError, OSError):
        return []


class TypingIngestPlugin(TargetPlugin):
    """Populate analytics.typedness and analytics.static_diagnostics.

    This plugin runs type checkers (pyright, pyrefly) and linters (ruff)
    to compute typedness scores and capture static diagnostics.

    Outputs
    -------
    - analytics.typedness: Type coverage metrics
    - analytics.static_diagnostics: Static analysis diagnostics
    """

    plugin_name: ClassVar[str] = "typing_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Populate analytics.typedness and analytics.static_diagnostics."
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute typing analysis.

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

        # Check if type checker is available (soft dependency)
        if ctx.resources.type_checker is None:
            log.info("Type checker not available, skipping typing analysis")
            return TargetResult.succeeded(row_counts={})

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
        tools = BuildToolAdapter(type_checker=ctx.resources.type_checker)

        # Execute step
        step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)
        result = await step.execute_async(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=str(ctx.repo_root),
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"Typing ingest failed: {errors}")

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = ["TypingIngestPlugin"]
