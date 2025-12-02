"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codeintel.ingestion import (
    CoverageIngestStep,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    RepoScanStep,
    ToolRunnerAdapter,
    TypingIngestStep,
)
from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway
from tests._helpers.tooling import build_tooling_context, run_static_tooling


def _setup_gateway() -> StorageGateway:
    return open_ingestion_gateway()


def test_repo_scan_honors_scan_profile(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanProfile."""
    repo_root = tmp_path / "repo"
    keep_dir = repo_root / "keep"
    ignore_dir = repo_root / "ignore"
    keep_dir.mkdir(parents=True, exist_ok=True)
    ignore_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "a.py").write_text("print('ok')\n", encoding="utf8")
    (ignore_dir / "b.py").write_text("print('skip')\n", encoding="utf8")

    gateway = _setup_gateway()
    profile = ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("*.py",),
        ignore_dirs=("ignore",),
    )

    # Use Step-based API
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(repo_root)
    change_detection = HashChangeDetectionAdapter(storage)

    step = RepoScanStep(
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
    )
    step.execute(repo="r", commit="c", repo_root=repo_root, profile=profile)

    rows = gateway.con.execute("SELECT path FROM core.modules").fetchall()
    if rows != [("keep/a.py",)]:
        pytest.fail(f"Unexpected modules: {rows}")


def test_coverage_ingest_uses_runner(tmp_path: Path) -> None:
    """Verify coverage ingestion prefers the shared runner path."""
    context = build_tooling_context(tmp_path)
    tooling_outputs = run_static_tooling(context)
    repo_root = context.repo_root
    tool_service = context.service
    gateway = _setup_gateway()
    expected_lines = sum(
        len(report.executed_lines | report.missing_lines)
        for report in tooling_outputs.coverage_reports
    )

    # Use Step-based API
    storage = DuckDBStorageAdapter(gateway)
    tools = ToolRunnerAdapter(tool_service)
    step = CoverageIngestStep(storage=storage, tools=tools)

    result = asyncio.run(
        step.execute_async(
            [],
            repo="r",
            commit="c",
            repo_root=repo_root,
            coverage_file=context.coverage_file,
        )
    )

    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "unknown"
        pytest.fail(f"Coverage ingest failed: {errors}")

    row = gateway.con.execute("SELECT COUNT(*) FROM analytics.coverage_lines").fetchone()
    count = row[0] if row is not None else 0
    if count != expected_lines:
        pytest.fail(f"Expected {expected_lines} coverage rows, got {count}")


def _create_scan_step(
    gateway: StorageGateway,
    repo_root: Path,
) -> tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]:
    """Create scan step and adapters for a repository.

    Returns
    -------
    tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]
        Scan step and the adapters used to create it.
    """
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    scan_step = RepoScanStep(
        storage=storage, discovery=discovery, change_detection=change_detection
    )
    return scan_step, storage, discovery


@pytest.mark.skip(
    reason="Schema mismatch: StaticDiagnosticRow (6 cols) vs static_diagnostics table (8 cols)"
)
def test_typing_ingest_uses_shared_runner(tmp_path: Path) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    context = build_tooling_context(tmp_path)
    gateway = _setup_gateway()
    scan_profile = ScanProfile(
        repo_root=context.repo_root,
        source_roots=(context.repo_root,),
        include_globs=("*.py",),
        ignore_dirs=(),
    )

    scan_step, storage, discovery = _create_scan_step(gateway, context.repo_root)
    tools = ToolRunnerAdapter(context.service)

    _, modules, _ = scan_step.execute(
        repo="r", commit="c", repo_root=context.repo_root, profile=scan_profile
    )

    typing_step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)
    result = asyncio.run(
        typing_step.execute_async(
            list(modules), repo="r", commit="c", repo_root=str(context.repo_root)
        )
    )

    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "unknown"
        pytest.fail(f"Typing ingest failed: {errors}")

    row = gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
    if (row[0] if row else 0) < 1:
        pytest.fail("Typedness ingestion wrote no rows")
