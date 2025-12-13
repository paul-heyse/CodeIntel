"""Tests ingest plugin.

This module provides `TestsIngestPlugin` that ingests pytest JSON reports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


TESTS_INGEST_METADATA = CorePluginMetadata(
    name="ingest.tests",
    version="3.0.0",
    description="Ingest pytest JSON reports.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="tests",
    provides=("analytics.test_results",),
    requires=("core.modules",),
    produces_tables=("analytics.test_results",),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


class TestsIngestPlugin(MetadataPlugin):
    """Ingest pytest JSON reports.

    This plugin reads pytest's JSON report output and extracts
    test results for storage.

    Outputs
    -------
    - analytics.test_results: Test execution results
    """

    _core_metadata: ClassVar[CorePluginMetadata] = TESTS_INGEST_METADATA

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
        _ = self

        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        report_path = resolve_report_file(ctx)
        if report_path is None:
            log.info("No pytest report found, skipping tests ingestion")
            return TargetResult.succeeded(row_counts={})

        storage = DuckDBStorageAdapter(ctx.gateway)

        step = TestsIngestStep(storage=storage)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
            json_report_path=report_path,
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            log.warning("Tests ingest failed: %s", errors)
            return TargetResult.failed(f"Tests ingest failed: {errors}")

        return TargetResult.succeeded(row_counts=result.table_counts or {})


def resolve_report_file(ctx: TargetExecutionContext) -> Path | None:
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
    candidates = [
        ctx.build_dir / "test-results" / "pytest-report.json",
        ctx.build_dir / "test-results" / "pytest_report.json",
        ctx.build_dir / "pytest-report.json",
        ctx.build_dir / "pytest_report.json",
        ctx.build_dir / "report.json",
        ctx.repo_root / "pytest-report.json",
        ctx.repo_root / "pytest_report.json",
        ctx.repo_root / "report.json",
        ctx.repo_root / "test-results" / "pytest-report.json",
        ctx.repo_root / ".pytest_cache" / "pytest_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


__all__ = [
    "TESTS_INGEST_METADATA",
    "TestsIngestPlugin",
    "get_module_paths",
    "paths_to_modules",
    "resolve_report_file",
]
