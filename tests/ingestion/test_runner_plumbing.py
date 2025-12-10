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
from tests._helpers.ingestion import ScanSetupOptions, make_scan_setup
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
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "keep/a.py": "print('ok')\n",
                "ignore/b.py": "print('skip')\n",
            },
            ignore_dirs=("ignore",),
        ),
    )

    try:
        setup.scan_step.execute(
            repo="r",
            commit="c",
            repo_root=setup.repo_root,
            profile=setup.profile,
        )

        rows = setup.gateway.con.table("core.modules").select("path").fetchall()
        if rows != [("keep/a.py",)]:
            pytest.fail(f"Unexpected modules: {rows}")
    finally:
        setup.gateway.close()


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
    scan_setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={"pkg/mod.py": "def add(x, y):\n    return x + y\n"},
            gateway_factory=GatewayFactory(),
        ),
    )
    gateway = scan_setup.gateway
    scan_step, storage, discovery = scan_setup.scan_step, scan_setup.storage, scan_setup.discovery
    tools = ToolRunnerAdapter(context.service)

    _, modules, _ = scan_step.execute(
        repo="r", commit="c", repo_root=scan_setup.repo_root, profile=scan_setup.profile
    )

    typing_step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)
    result = asyncio.run(
        typing_step.execute_async(
            list(modules), repo="r", commit="c", repo_root=str(scan_setup.repo_root)
        )
    )

    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "unknown"
        pytest.fail(f"Typing ingest failed: {errors}")

    row = gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
    if (row[0] if row else 0) < 1:
        pytest.fail("Typedness ingestion wrote no rows")
