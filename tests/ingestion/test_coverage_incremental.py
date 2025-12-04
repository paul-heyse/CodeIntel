"""Coverage ingest incremental harness tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codeintel.config import ConfigBuilder
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion import (
    CoverageIngestStep,
    DuckDBStorageAdapter,
    IngestExecutionContext,
    ToolRunnerAdapter,
)
from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    default_code_profile,
    default_config_profile,
)
from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
from codeintel.ingestion.plugins import (
    IngestRuntimeScratch,
    get_ingest_registry,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.resources import (
    ModuleProvider,
    ResourceRegistry,
    ToolsProvider,
    TrackerConfig,
    TrackerProvider,
)
from codeintel.ingestion.tool_service import ToolService
from codeintel.ingestion.tools.results import CoverageFileSummary, CoverageReport
from tests._helpers.gateway import open_ingestion_gateway


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


def _build_plugin_context(
    repo_root: Path,
    *,
    tools: ToolsConfig | None = None,
    scratch: IngestRuntimeScratch | None = None,
    change_tracker: ChangeTracker | None = None,
) -> IngestExecutionContext:
    """Construct a minimal plugin context for coverage tests.

    Parameters
    ----------
    repo_root
        Repository root path.
    tools
        Optional tools configuration.
    scratch
        Optional shared scratch space.
    change_tracker
        Optional change tracker for incremental ingestion.

    Returns
    -------
    IngestExecutionContext
        Context populated with snapshot, paths, gateway, and profiles.
    """
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root)
    gateway = open_ingestion_gateway()
    effective_tools = tools or ToolsConfig.default()
    code_profile = default_code_profile(repo_root)
    effective_scratch = scratch if scratch is not None else IngestRuntimeScratch()
    if change_tracker is not None:
        effective_scratch.declare("change_tracker", change_tracker)

    # Build resource registry
    registry = ResourceRegistry()
    tracker_config = TrackerConfig(scratch=effective_scratch, profile=code_profile)
    registry.register(TrackerProvider, TrackerProvider(gateway, snapshot, tracker_config))
    registry.register(ToolsProvider, ToolsProvider(effective_tools, paths.tool_cache))
    registry.register(ModuleProvider, ModuleProvider(gateway, snapshot, profile=code_profile))

    return IngestExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=effective_tools,
        code_profile=code_profile,
        config_profile=default_config_profile(repo_root),
        scratch=effective_scratch,
        resources=registry,
    )


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

    gateway = open_ingestion_gateway()
    try:
        snapshot = SnapshotRef.from_args(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
        paths = BuildPaths.from_repo_root(repo_root)
        tools = ToolsConfig.default()
        code_profile = default_code_profile(repo_root)
        scratch = IngestRuntimeScratch()

        # Create a tracker and store in scratch
        tracker = ChangeTracker(
            gateway=gateway,
            change_request=ChangeRequest(
                repo=snapshot.repo,
                commit=snapshot.commit,
                repo_root=repo_root,
                modules=(),
            ),
            modules=(),
            change_set=ChangeSet(added=[], modified=[], deleted=[]),
            policy=IncrementalIngestPolicy(),
        )
        scratch.declare("change_tracker", tracker)

        # Build resource registry
        registry = ResourceRegistry()
        tracker_config = TrackerConfig(scratch=scratch, profile=code_profile)
        registry.register(TrackerProvider, TrackerProvider(gateway, snapshot, tracker_config))
        registry.register(ToolsProvider, ToolsProvider(tools, paths.tool_cache))
        registry.register(ModuleProvider, ModuleProvider(gateway, snapshot, profile=code_profile))

        ctx = IngestExecutionContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=tools,
            code_profile=code_profile,
            config_profile=default_config_profile(repo_root),
            scratch=scratch,
            resources=registry,
        )

        plugin_registry = get_ingest_registry()
        coverage_plugin = plugin_registry.get("coverage_ingest")
        result = coverage_plugin.execute(ctx)

        # Should succeed (or skip if no coverage data)
        if not result.success and not result.skipped:
            pytest.fail(f"coverage_ingest failed unexpectedly: {result.error}")
    finally:
        gateway.close()


def test_coverage_plugin_without_coverage_file_skips(tmp_path: Path) -> None:
    """Coverage plugin should skip gracefully without a coverage file."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    ctx = _build_plugin_context(repo_root)
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
