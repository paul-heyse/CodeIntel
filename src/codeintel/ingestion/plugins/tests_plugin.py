"""Tests ingest plugin.

This module provides `TestsIngestPlugin` that ingests pytest JSON reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginKind, PluginMetadata, PluginStage
from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules

if TYPE_CHECKING:
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
    _core_metadata: ClassVar[CorePluginMetadata] = TESTS_INGEST_METADATA

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
        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        # Get report path - check common locations
        report_path = resolve_report_file(ctx)
        if report_path is None:
            log.info("No pytest report found, skipping tests ingestion")
            return TargetResult.succeeded(row_counts={})

        storage = DuckDBStorageAdapter(ctx.gateway)
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


__all__ = [
    "TESTS_INGEST_METADATA",
    "TestsIngestPlugin",
    "get_module_paths",
    "paths_to_modules",
    "resolve_report_file",
]
