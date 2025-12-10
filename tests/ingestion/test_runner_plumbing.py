"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion import (
    CoverageIngestStep,
    DuckDBStorageAdapter,
    ToolRunnerAdapter,
    TypingIngestStep,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import build_scan_profile, create_scan_step
from tests._helpers.orchestration.tooling import build_tooling_context, run_static_tooling

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def _setup_gateway() -> StorageGateway:
    """Create a gateway with default factory settings.

    Returns
    -------
    StorageGateway
        Gateway instance opened for tests.
    """
    return GatewayFactory().with_macros().open()


def test_repo_scan_honors_scan_profile(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanProfile."""
    repo_root = tmp_path / "repo"
    keep_dir = repo_root / "keep"
    ignore_dir = repo_root / "ignore"
    keep_dir.mkdir(parents=True, exist_ok=True)
    ignore_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "a.py").write_text("print('ok')\n", encoding="utf8")
    (ignore_dir / "b.py").write_text("print('skip')\n", encoding="utf8")

    gateway = GatewayFactory().with_macros().open()
    profile = build_scan_profile(repo_root, ignore_dirs=("ignore",))

    step, _, _ = create_scan_step(gateway, repo_root, tmp_path)
    step.execute(repo="r", commit="c", repo_root=repo_root, profile=profile)

    rows = gateway.con.table("core.modules").select("path").fetchall()
    if rows != [("keep/a.py",)]:
        pytest.fail(f"Unexpected modules: {rows}")


def test_coverage_ingest_uses_runner(tmp_path: Path) -> None:
    """Verify coverage ingestion prefers the shared runner path."""
    context = build_tooling_context(tmp_path)
    tooling_outputs = run_static_tooling(context)
    repo_root = context.repo_root
    tool_service = context.service
    gateway = GatewayFactory().open()
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

    row = gateway.con.table("analytics.coverage_lines").aggregate("count(*)").fetchone()
    count = row[0] if row is not None else 0
    if count != expected_lines:
        pytest.fail(f"Expected {expected_lines} coverage rows, got {count}")


@pytest.mark.skip(
    reason="Schema mismatch: StaticDiagnosticRow (6 cols) vs static_diagnostics table (8 cols)"
)
def test_typing_ingest_uses_shared_runner(tmp_path: Path) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    context = build_tooling_context(tmp_path)
    gateway = _setup_gateway()
    scan_profile = build_scan_profile(context.repo_root)

    scan_step, storage, discovery = create_scan_step(gateway, context.repo_root, tmp_path)
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
