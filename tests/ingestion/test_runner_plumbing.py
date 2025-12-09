"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

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
from codeintel.ingestion.infrastructure.scanning import ScanProfile
from tests._helpers import build_repo_tree
from tests._helpers.gateway import GatewayFactory
from tests._helpers.orchestration.tooling import build_tooling_context, run_static_tooling

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def build_repo_with_configs(
    tmp_path: Path,
    *,
    include_invalid: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    """Create a realistic repo layout with Python modules and config files.

    Returns
    -------
    tuple[Path, tuple[str, ...]]
        Repository root path and the relative module paths created.
    """
    structure: dict[str, str] = {
        "pkg/__init__.py": "",
        "pkg/service.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
        "config/app.yaml": "service:\n  retries: 3\n  hosts:\n    - api.local\n",
        "config/settings.toml": 'feature = true\n[db]\nurl = "duckdb:///tmp.db"\n',
        "config/app.ini": "[service]\ntimeout=30\n",
    }
    if include_invalid:
        structure["config/broken.yml"] = ":\n  - invalid\n"

    repo_root = build_repo_tree(tmp_path / "repo", structure)
    modules = tuple(path for path in structure if path.endswith(".py"))
    return repo_root, modules


def _setup_gateway() -> StorageGateway:
    """Create a gateway with default factory settings.

    Returns
    -------
    StorageGateway
        Gateway instance opened for tests.
    """
    return GatewayFactory().open()


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

    rows = gateway.con.table("core.modules").select("path").fetchall()
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

    row = gateway.con.table("analytics.coverage_lines").aggregate("count(*)").fetchone()
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
