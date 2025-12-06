"""Tests ingest plugin.

This module provides `TestsIngestPlugin` that ingests pytest JSON reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
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


class TestsIngestPlugin(TargetPlugin):
    """Ingest pytest JSON reports.

    This plugin reads pytest's JSON report output and extracts
    test results for storage.

    Outputs
    -------
    - analytics.test_results: Test execution results
    """

    plugin_name: ClassVar[str] = "tests_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Ingest pytest JSON reports."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute tests ingestion.

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

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

        # Create storage adapter
        storage = DuckDBStorageAdapter(ctx.gateway)

        # Get report path - check common locations
        report_path = _resolve_report_file(ctx)
        if report_path is None:
            log.info("No pytest report found, skipping tests ingestion")
            return TargetResult.succeeded(row_counts={})

        # Execute step
        step = TestsIngestStep(storage=storage)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
            json_report_path=report_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"Tests ingest failed: {errors}")

        return TargetResult.succeeded(row_counts=result.table_counts or {})


def _resolve_report_file(ctx: TargetExecutionContext) -> Path | None:
    """Resolve the pytest report file.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    Path | None
        Path to report file or None if not found.
    """
    # Check common locations - including various naming conventions
    candidates = [
        # Standard locations
        ctx.build_dir / "test-results" / "pytest-report.json",
        ctx.build_dir / "test-results" / "pytest_report.json",
        ctx.build_dir / "pytest-report.json",
        ctx.build_dir / "pytest_report.json",
        ctx.build_dir / "report.json",
        # Repo root locations
        ctx.repo_root / "pytest-report.json",
        ctx.repo_root / "pytest_report.json",
        ctx.repo_root / "report.json",
        # CI common locations
        ctx.repo_root / "test-results" / "pytest-report.json",
        ctx.repo_root / ".pytest_cache" / "pytest_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


__all__ = ["TestsIngestPlugin"]
