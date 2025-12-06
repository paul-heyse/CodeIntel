"""SCIP ingest plugin.

This module provides `ScipIngestPlugin` that runs scip-python
and persists symbols and GOID crosswalk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.scip_ingest import ScipIngestConfig, ScipIngestStep
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


def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
    """Compute row counts for output tables.

    Returns
    -------
    dict[str, int]
        Row counts per table.
    """
    row_counts: dict[str, int] = {}
    for table_key in ctx.contract.table_keys:
        try:
            count = ctx.gateway.con.execute(
                f"SELECT COUNT(*) FROM {table_key} "  # noqa: S608
                f"WHERE repo = ? AND commit = ?",
                [ctx.repo, ctx.commit],
            ).fetchone()
            row_counts[table_key] = int(count[0]) if count else 0
        except (RuntimeError, OSError):
            row_counts[table_key] = 0
    return row_counts


class ScipIngestPlugin(TargetPlugin):
    """Run scip-python and persist symbols and GOID crosswalk.

    This plugin executes the SCIP-Python indexer to generate semantic
    code intelligence data, including symbol information and global
    identifier crosswalk.

    Outputs
    -------
    - index.scip: SCIP index file
    - core.scip_symbols: Symbol table
    - core.goid_crosswalk: GOID crosswalk
    """

    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Run scip-python and persist symbols and GOID crosswalk."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute SCIP indexing.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.

        Raises
        ------
        ToolNotAvailableError
            When the scip-python tool is not available.
        """
        _ = self  # Protocol method requires instance

        # Check tool availability
        if ctx.resources.scip_indexer is None:
            raise ToolNotAvailableError(target=self.plugin_name, tool="scip-python")

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(scip_indexer=ctx.resources.scip_indexer)

        # Create config
        scip_dir = ctx.scip_dir
        config = ScipIngestConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.json",
        )

        # Execute step
        step = ScipIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(modules, config)

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"SCIP ingest failed: {errors}")

        # Compute row counts
        row_counts = _compute_row_counts(ctx)
        return TargetResult.succeeded(
            row_counts=row_counts,
            artifacts_written=["index.scip", "index.json"],
        )


__all__ = ["ScipIngestPlugin"]
