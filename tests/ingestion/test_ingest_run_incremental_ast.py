"""Verify AST plugin execution and result handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    default_code_profile,
    default_config_profile,
)
from codeintel.ingestion.plugins import (
    IngestPluginContext,
    IngestPluginResult,
    IngestRuntimeScratch,
    get_ingest_registry,
)
from tests._helpers.gateway import open_ingestion_gateway


def test_ast_extract_plugin_succeeds_with_tracker(tmp_path: Path) -> None:
    """Ensure AST plugin executes successfully when change_tracker is available."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")
    (src_dir / "b.py").write_text("print('b')\n", encoding="utf8")

    snapshot = SnapshotRef.from_args(repo="demo/ast", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root)
    gateway = open_ingestion_gateway()
    scratch = IngestRuntimeScratch()

    try:
        registry = get_ingest_registry()

        # First run repo_scan to get change_tracker
        repo_scan_ctx = IngestPluginContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=ToolsConfig.default(),
            code_profile=default_code_profile(repo_root),
            config_profile=default_config_profile(repo_root),
            scratch=scratch,
        )

        repo_scan = registry.get("repo_scan")
        scan_result = repo_scan.execute(repo_scan_ctx)

        if not scan_result.success:
            pytest.fail(f"repo_scan failed: {scan_result.error}")

        # Get change_tracker from scratch
        change_tracker = scratch.consume("change_tracker")
        if change_tracker is None:
            pytest.fail("repo_scan did not populate change_tracker")

        # Now run ast_extract with change_tracker
        ast_ctx = IngestPluginContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=ToolsConfig.default(),
            code_profile=default_code_profile(repo_root),
            config_profile=default_config_profile(repo_root),
            change_tracker=change_tracker,  # type: ignore[arg-type]
            scratch=scratch,
        )

        ast_plugin = registry.get("ast_extract")
        ast_result = ast_plugin.execute(ast_ctx)

        if not ast_result.success:
            pytest.fail(f"ast_extract failed: {ast_result.error}")

        # Check that row counts were returned
        if ast_result.row_counts is None:
            pytest.fail("ast_extract should return row_counts")

        # Verify tables were populated
        core_tables = {"core.ast_nodes", "core.ast_metrics"}
        if not any(table in (ast_result.row_counts or {}) for table in core_tables):
            pytest.fail(f"Expected some of {core_tables} in row_counts: {ast_result.row_counts}")

    finally:
        gateway.close()


def test_ast_extract_plugin_fails_without_tracker(tmp_path: Path) -> None:
    """AST plugin should return failure when change_tracker is missing."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")

    snapshot = SnapshotRef.from_args(repo="demo/ast", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root)
    gateway = open_ingestion_gateway()

    try:
        registry = get_ingest_registry()

        # Run ast_extract without change_tracker
        ctx = IngestPluginContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=ToolsConfig.default(),
            code_profile=default_code_profile(repo_root),
            config_profile=default_config_profile(repo_root),
            change_tracker=None,  # Missing!
            scratch=IngestRuntimeScratch(),
        )

        ast_plugin = registry.get("ast_extract")
        result = ast_plugin.execute(ctx)

        if result.success:
            pytest.fail("ast_extract should fail without change_tracker")
        if result.error is None:
            pytest.fail("error message should be set on failure")
        if result.error_kind is None:
            pytest.fail("error_kind should be set on failure")

    finally:
        gateway.close()


def test_plugin_result_ok_with_row_counts() -> None:
    """Verify ok result properly captures row counts."""
    expected_ast_nodes = 100
    expected_ast_metrics = 50
    row_counts = {"core.ast_nodes": expected_ast_nodes, "core.ast_metrics": expected_ast_metrics}
    result = IngestPluginResult.ok(row_counts=row_counts)

    if not result.success:
        pytest.fail("ok result should have success=True")
    if result.row_counts is None:
        pytest.fail("row_counts should be set")
    if result.row_counts.get("core.ast_nodes") != expected_ast_nodes:
        pytest.fail(f"Unexpected row_counts: {result.row_counts}")


def test_cst_extract_plugin_succeeds_with_tracker(tmp_path: Path) -> None:
    """Ensure CST plugin executes successfully when change_tracker is available."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "mod.py").write_text("x = 1\n", encoding="utf8")

    snapshot = SnapshotRef.from_args(repo="demo/cst", commit="xyz789", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root)
    gateway = open_ingestion_gateway()
    scratch = IngestRuntimeScratch()

    try:
        registry = get_ingest_registry()

        # First run repo_scan
        repo_scan_ctx = IngestPluginContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=ToolsConfig.default(),
            code_profile=default_code_profile(repo_root),
            config_profile=default_config_profile(repo_root),
            scratch=scratch,
        )

        repo_scan = registry.get("repo_scan")
        scan_result = repo_scan.execute(repo_scan_ctx)

        if not scan_result.success:
            pytest.fail(f"repo_scan failed: {scan_result.error}")

        # Get change_tracker
        change_tracker = scratch.consume("change_tracker")
        if change_tracker is None:
            pytest.fail("repo_scan did not populate change_tracker")

        # Run cst_extract
        cst_ctx = IngestPluginContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=ToolsConfig.default(),
            code_profile=default_code_profile(repo_root),
            config_profile=default_config_profile(repo_root),
            change_tracker=change_tracker,  # type: ignore[arg-type]
            scratch=scratch,
        )

        cst_plugin = registry.get("cst_extract")
        cst_result = cst_plugin.execute(cst_ctx)

        if not cst_result.success:
            pytest.fail(f"cst_extract failed: {cst_result.error}")

    finally:
        gateway.close()
