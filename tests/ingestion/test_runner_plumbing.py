"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import (
    CoverageIngestConfig,
    RepoScanConfig,
    ToolsConfig,
    TypingIngestConfig,
)
from codeintel.ingestion.coverage_ingest import ingest_coverage_lines
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.typing_ingest import ingest_typing_signals
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import open_ingestion_gateway
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
    cfg = RepoScanConfig(repo_root=repo_root, repo="r", commit="c")
    profile = ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("*.py",),
        ignore_dirs=("ignore",),
    )
    ingest_repo(gateway, cfg=cfg, code_profile=profile)

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
    cfg = CoverageIngestConfig(
        repo_root=repo_root,
        repo="r",
        commit="c",
        coverage_file=context.coverage_file,
    )
    ingest_coverage_lines(
        gateway,
        cfg=cfg,
        tools=ToolsConfig.model_validate({}),
        tool_service=tool_service,
    )

    row = gateway.con.execute("SELECT COUNT(*) FROM analytics.coverage_lines").fetchone()
    count = row[0] if row is not None else 0
    if count != expected_lines:
        pytest.fail(f"Expected {expected_lines} coverage rows, got {count}")


def test_typing_ingest_uses_shared_runner(tmp_path: Path) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    context = build_tooling_context(tmp_path)
    repo_root = context.repo_root
    tool_service = context.service

    gateway = _setup_gateway()
    cfg = TypingIngestConfig(repo_root=repo_root, repo="r", commit="c")
    ingest_typing_signals(
        gateway,
        cfg=cfg,
        tool_service=tool_service,
        tools=ToolsConfig.model_validate({}),
    )

    row = gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
    typedness_rows = row[0] if row is not None else 0
    if typedness_rows < 1:
        pytest.fail("Typedness ingestion wrote no rows")
