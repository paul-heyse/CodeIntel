"""Coverage ingest incremental harness tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codeintel.config import ConfigBuilder
from codeintel.config.models import ToolsConfig
from codeintel.ingestion import (
    CoverageIngestStep,
    DuckDBStorageAdapter,
    IngestExecutionContext,
    ToolRunnerAdapter,
)
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.results import CoverageFileSummary, CoverageReport
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.plugins import (
    get_ingest_registry,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.tracker import ChangeTracker, IncrementalIngestPolicy
from tests._helpers.gateway import open_ingestion_gateway
from tests._helpers.harnesses import IngestTestSetup


class _FakeCoverageService(ToolService):
    """Provide synthetic coverage reports without invoking external tools."""

    def __init__(self, report: CoverageFileSummary, repo_root: Path) -> None:
        tools_cfg = ToolsConfig.default()
        runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache", tools_config=tools_cfg)
        super().__init__(runner, tools_cfg)
        self._report = report

    async def run_coverage_report(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageReport:
        del repo_root, coverage_file, output_path
        return CoverageReport(files=(self._report,))


def _build_plugin_context(repo_root: Path) -> tuple[IngestExecutionContext, IngestTestSetup]:
    """Construct a minimal plugin context for coverage tests.

    Parameters
    ----------
    repo_root
        Repository root path.

    Returns
    -------
    tuple[IngestExecutionContext, IngestTestSetup]
        Context and setup bundle.
    """
    gateway = open_ingestion_gateway()
    setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
    ctx = setup.build_context("coverage_test")
    return ctx, setup


def test_coverage_ingest_runs_full_rebuild_with_tracker(tmp_path: Path) -> None:
    """Ensure coverage ingest performs a full rebuild when invoked via tracker."""
    gateway = open_ingestion_gateway()
    try:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        coverage_file = repo_root / ".coverage"
        coverage_file.touch()

        builder = ConfigBuilder.from_snapshot(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=repo_root,
            build_dir=repo_root / "build",
        )
        cfg = builder.coverage_ingest(coverage_file=coverage_file)
        report = CoverageFileSummary(
            rel_path="pkg/mod.py",
            executed_lines=frozenset({1, 2}),
            missing_lines=frozenset({3}),
        )
        fake_service = _FakeCoverageService(report, repo_root)

        # Use the new Step-based API
        storage = DuckDBStorageAdapter(gateway)
        tools = ToolRunnerAdapter(fake_service)
        step = CoverageIngestStep(storage=storage, tools=tools)

        result = asyncio.run(
            step.execute_async(
                [],  # modules not used for coverage
                repo=cfg.repo,
                commit=cfg.commit,
                repo_root=repo_root,
                coverage_file=coverage_file,
            )
        )

        if not result.success:
            pytest.fail(f"Coverage ingest failed: {'; '.join(result.errors)}")

        count = gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.coverage_lines
            WHERE repo = ? AND commit = ?
            """,
            [cfg.repo, cfg.commit],
        ).fetchone()
        if count is None:
            pytest.fail("Expected coverage rows but query returned no result")
        expected = len(report.executed_lines | report.missing_lines)
        count_value = count[0]
        if count_value != expected:
            pytest.fail(f"Expected {expected} coverage rows, found {count_value}")
    finally:
        gateway.close()


def test_coverage_plugin_executes_with_tracker(tmp_path: Path) -> None:
    """Ensure coverage plugin can execute when tracker is available."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # Create a minimal coverage file structure
    coverage_file = repo_root / ".coverage"
    coverage_file.touch()

    ctx, setup = _build_plugin_context(repo_root)
    try:
        # Create a tracker and store in scratch
        tracker = ChangeTracker(
            gateway=setup.gateway,
            change_request=ChangeRequest(
                repo=setup.snapshot.repo,
                commit=setup.snapshot.commit,
                repo_root=repo_root,
                modules=(),
            ),
            modules=(),
            change_set=ChangeSet(added=[], modified=[], deleted=[]),
            policy=IncrementalIngestPolicy(),
        )
        setup.scratch.declare("change_tracker", tracker)

        plugin_registry = get_ingest_registry()
        coverage_plugin = plugin_registry.get("coverage_ingest")
        result = coverage_plugin.execute(ctx)

        # Should succeed (or skip if no coverage data)
        if not result.success and not result.skipped:
            pytest.fail(f"coverage_ingest failed unexpectedly: {result.error}")
    finally:
        ctx.gateway.close()


def test_coverage_plugin_without_coverage_file_skips(tmp_path: Path) -> None:
    """Coverage plugin should skip gracefully without a coverage file."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    ctx, _setup = _build_plugin_context(repo_root)
    try:
        registry = get_ingest_registry()
        coverage_plugin = registry.get("coverage_ingest")

        # Execute without coverage file
        result = coverage_plugin.execute(ctx)

        # Plugin should succeed but may skip if coverage file is missing
        # The plugin doesn't require change_tracker if no coverage file exists
        if not result.success and not result.skipped:
            pytest.fail(f"coverage_ingest failed unexpectedly: {result.error}")
    finally:
        ctx.gateway.close()
