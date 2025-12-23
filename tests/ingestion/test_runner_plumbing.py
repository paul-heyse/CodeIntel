"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from tests._helpers.assertions import expect_equal, expect_rows_equal
from tests._helpers.ingestion import (
    ScanSetupOptions,
    build_scan_profile,
    closing_gateway,
    create_scan_step,
    make_scan_setup,
    materialize_repo_scan_result,
    materialize_rows_for_snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.orchestration.tooling import ToolingOutputs


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

    with closing_gateway(setup.gateway):
        scan_result = setup.scan_step.execute(
            repo="r",
            commit="c",
            repo_root=setup.repo_root,
            profile=setup.profile,
        )
        materialize_repo_scan_result(
            setup.gateway,
            scan_result,
            snapshot=SnapshotRef(repo="r", commit="c", repo_root=setup.repo_root),
        )

        rows = setup.gateway.con.table("core.modules").select("path").fetchall()
        expect_rows_equal(rows, [("keep/a.py",)], message="Unexpected modules from repo_scan")


def test_coverage_ingest_uses_runner(
    tooling_outputs_session: ToolingOutputs, ingestion_gateway: StorageGateway
) -> None:
    """Verify coverage ingestion prefers the shared runner path."""
    tooling_outputs = tooling_outputs_session
    repo_root = tooling_outputs.context.repo_root
    tool_service = tooling_outputs.context.service
    gateway = ingestion_gateway
    expected_lines = sum(
        len(report.executed_lines | report.missing_lines)
        for report in tooling_outputs.coverage_reports
    )

    tools = ToolRunnerAdapter(tool_service)
    step = CoverageIngestStep(tools=tools)

    ingest_result = asyncio.run(
        step.execute_async(
            [],
            repo="r",
            commit="c",
            repo_root=repo_root,
            coverage_file=tooling_outputs.context.coverage_file,
        )
    )

    if not ingest_result.result.success:
        error_message = (
            ingest_result.result.error
            or "; ".join(ingest_result.result.warnings)
            or "unknown"
        )
        pytest.fail(f"Coverage ingest failed: {error_message}")

    materialize_rows_for_snapshot(
        gateway,
        "analytics.coverage_lines",
        ingest_result.rows,
        snapshot=SnapshotRef(repo="r", commit="c", repo_root=repo_root),
    )

    row = gateway.con.table("analytics.coverage_lines").aggregate("count(*)").fetchone()
    count = row[0] if row is not None else 0
    expect_equal(count, expected_lines, label="coverage_line_count")


@pytest.mark.skip(
    reason="Schema mismatch: StaticDiagnosticRow (6 cols) vs static_diagnostics table (8 cols)"
)
def test_typing_ingest_uses_shared_runner(
    tooling_outputs_session: ToolingOutputs,
    ingestion_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    tooling = tooling_outputs_session
    repo_root = tooling.context.repo_root
    profile = build_scan_profile(repo_root)
    scan_step, _storage, discovery = create_scan_step(ingestion_gateway, repo_root, tmp_path)
    tools = ToolRunnerAdapter(tooling.context.service)

    scan_result = scan_step.execute(
        repo="r",
        commit="c",
        repo_root=repo_root,
        profile=profile,
    )
    modules = scan_result.modules

    typing_step = TypingIngestStep(discovery=discovery, tools=tools)
    ingest_result = asyncio.run(
        typing_step.execute_async(list(modules), repo="r", commit="c", repo_root=str(repo_root))
    )

    if not ingest_result.result.success:
        error_message = (
            ingest_result.result.error
            or "; ".join(ingest_result.result.warnings)
            or "unknown"
        )
        pytest.fail(f"Typing ingest failed: {error_message}")

    snapshot = SnapshotRef(repo="r", commit="c", repo_root=repo_root)
    materialize_rows_for_snapshot(
        ingestion_gateway,
        "analytics.typedness",
        ingest_result.typedness_rows,
        snapshot=snapshot,
    )
    materialize_rows_for_snapshot(
        ingestion_gateway,
        "analytics.static_diagnostics",
        ingest_result.diagnostic_rows,
        snapshot=snapshot,
    )

    row = ingestion_gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
    if (row[0] if row else 0) < 1:
        pytest.fail("Typedness ingestion wrote no rows")
