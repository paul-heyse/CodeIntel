"""Tests for ingestion run reporting and error classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion import IngestExecutionContext
from codeintel.ingestion.core.runs import IngestRun, IngestRunSink
from codeintel.ingestion.plugins import (
    IngestPluginResult,
    IngestRuntimeScratch,
    get_ingest_registry,
)
from codeintel.ingestion.resources import (
    ModuleProvider,
    ResourceRegistry,
    ToolsProvider,
    TrackerConfig,
    TrackerProvider,
)
from codeintel.ingestion.utilities.scanning import (
    default_code_profile,
    default_config_profile,
)
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway


@dataclass
class RecordingSink(IngestRunSink):
    """Simple sink for capturing run records in-memory."""

    runs: list[IngestRun]

    def record(self, run: IngestRun) -> None:
        """Store the provided run for later inspection."""
        self.runs.append(run)


def _build_plugin_context(
    tmp_path: Path,
    *,
    scratch: IngestRuntimeScratch | None = None,
) -> tuple[IngestExecutionContext, SnapshotRef]:
    """Build a plugin context for testing.

    Parameters
    ----------
    tmp_path
        Temp directory for test repo.
    scratch
        Optional shared scratch space.

    Returns
    -------
    tuple[IngestExecutionContext, SnapshotRef]
        Plugin context and snapshot reference.
    """
    repo_root = tmp_path / "repo"
    if not repo_root.exists():
        repo_root.mkdir(parents=True)
    paths = BuildPaths.from_repo_root(repo_root)
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="abc123", repo_root=repo_root)
    gateway = open_ingestion_gateway()
    tools = ToolsConfig.default()
    code_profile = default_code_profile(repo_root)

    actual_scratch = scratch if scratch is not None else IngestRuntimeScratch()

    # Build resource registry with providers
    registry = ResourceRegistry()
    tracker_config = TrackerConfig(scratch=actual_scratch, profile=code_profile)
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
        scratch=actual_scratch,
        resources=registry,
    )
    return ctx, snapshot


def test_plugin_execution_success(tmp_path: Path) -> None:
    """Ensure plugins can be executed successfully and return proper results."""
    repo_root = tmp_path / "repo" / "src" / "pkg"
    repo_root.mkdir(parents=True)
    (repo_root / "a.py").write_text('"""docstring"""\n', encoding="utf8")

    scratch = IngestRuntimeScratch()
    ctx, _ = _build_plugin_context(tmp_path, scratch=scratch)

    try:
        registry = get_ingest_registry()

        # Execute repo_scan
        repo_scan_plugin = registry.get("repo_scan")
        result = repo_scan_plugin.execute(ctx)

        if not result.success:
            pytest.fail(f"repo_scan failed: {result.error}")

        # Get change_tracker from scratch for downstream plugins
        change_tracker = scratch.consume("change_tracker")
        if change_tracker is None:
            pytest.fail("repo_scan did not populate change_tracker in scratch")

        # Execute docstrings_ingest with change_tracker available
        ctx2 = IngestExecutionContext(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tools=ctx.tools,
            code_profile=ctx.code_profile,
            config_profile=ctx.config_profile,
            resources=ctx.resources,  # Reuse resources from ctx
            scratch=scratch,
        )
        # Store change_tracker in scratch for plugins that need it
        ctx2.scratch.declare("change_tracker", change_tracker)

        docstrings_plugin = registry.get("docstrings_ingest")
        doc_result = docstrings_plugin.execute(ctx2)

        if not doc_result.success:
            pytest.fail(f"docstrings_ingest failed: {doc_result.error}")

        # Check row counts were returned
        if doc_result.row_counts is None:
            pytest.fail("row_counts should be populated")

    finally:
        ctx.gateway.close()


def test_plugin_execution_succeeds_without_tracker_in_scratch(tmp_path: Path) -> None:
    """Verify plugin succeeds when tracker is missing from scratch but modules exist in DB."""
    repo_root = tmp_path / "repo" / "src" / "pkg"
    repo_root.mkdir(parents=True)
    # Valid Python file for repo_scan
    (repo_root / "a.py").write_text("x = 1\n", encoding="utf8")

    scratch = IngestRuntimeScratch()
    ctx, _ = _build_plugin_context(tmp_path, scratch=scratch)

    try:
        registry = get_ingest_registry()

        # Execute repo_scan first to populate modules in DB
        repo_scan_plugin = registry.get("repo_scan")
        repo_scan_plugin.execute(ctx)

        # Execute cst_extract without change_tracker in scratch
        # New behavior: plugin reads modules from DB when tracker not in scratch
        cst_plugin = registry.get("cst_extract")

        # Create context without change_tracker in scratch
        ctx2 = IngestExecutionContext(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tools=ctx.tools,
            code_profile=ctx.code_profile,
            config_profile=ctx.config_profile,
            resources=ctx.resources,  # Reuse resources from ctx
            scratch=IngestRuntimeScratch(),  # Fresh scratch without change_tracker
        )

        result = cst_plugin.execute(ctx2)

        # Plugin should succeed - it reads modules from DB when tracker is missing
        if not result.success:
            pytest.fail(f"cst_extract should succeed: {result.error}")

        # Should have processed the module from repo_scan
        total_rows = sum(result.row_counts.values()) if result.row_counts else 0
        if total_rows == 0:
            pytest.fail("Expected some rows, plugin read modules from DB")

    finally:
        ctx.gateway.close()


def test_plugin_skip_result() -> None:
    """Verify skip results are properly constructed."""
    result = IngestPluginResult.skip("Missing required tool")

    if not result.success:
        pytest.fail("Skip should have success=True")
    if not result.skipped:
        pytest.fail("Skip should have skipped=True")
    if result.skip_reason != "Missing required tool":
        pytest.fail(f"Unexpected skip_reason: {result.skip_reason}")


def test_plugin_fail_result() -> None:
    """Verify failure results are properly constructed."""
    result = IngestPluginResult.fail("Something went wrong", error_kind="ValueError")

    if result.success:
        pytest.fail("Fail should have success=False")
    if result.error != "Something went wrong":
        pytest.fail(f"Unexpected error: {result.error}")
    if result.error_kind != "ValueError":
        pytest.fail(f"Unexpected error_kind: {result.error_kind}")
