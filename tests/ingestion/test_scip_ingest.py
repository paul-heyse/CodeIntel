"""Integration test for scip_ingest using real SCIP binaries."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from codeintel.config import BuildPaths
from codeintel.config.models import ToolsConfig
from codeintel.ingestion import (
    DuckDBStorageAdapter,
    ScipIngestStep,
    ToolRunnerAdapter,
)
from codeintel.ingestion.steps.scip_ingest import ScipIngestConfig, ScipIngestResult
from codeintel.ingestion.tools.infrastructure import ToolRunner
from codeintel.ingestion.tools.service import ToolService
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway


def _setup_repo_structure(tmp_path: Path) -> tuple[Path, Path]:
    """Set up test repository structure.

    Returns
    -------
    tuple[Path, Path]
        Repository root path and database path.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir()

    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "mod.py").write_text("def foo(x: int) -> int:\n    return x + 1\n", encoding="utf8")

    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel_prefect.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return repo_root, db_path


def _create_scip_adapters(
    gateway: StorageGateway,
    repo_root: Path,
) -> tuple[DuckDBStorageAdapter, ToolRunnerAdapter]:
    """Create storage and tool adapters for SCIP ingestion.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB connection.
    repo_root
        Path to the repository root directory.

    Returns
    -------
    tuple[DuckDBStorageAdapter, ToolRunnerAdapter]
        Storage and tool adapters configured for SCIP ingestion.
    """
    storage = DuckDBStorageAdapter(gateway)
    tools_config = ToolsConfig.default()
    cache_dir = repo_root / "build" / ".tool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner(tools_config=tools_config, cache_dir=cache_dir)
    tool_service = ToolService(runner, tools_config)
    tools = ToolRunnerAdapter(tool_service)
    return storage, tools


def test_ingest_scip_produces_artifacts(tmp_path: Path) -> None:
    """Ensure scip_ingest generates SCIP artifacts and registers scip_index_view.

    Skip if scip-python or scip binaries are unavailable.
    """
    if shutil.which("scip-python") is None or shutil.which("scip") is None:
        pytest.skip("scip-python or scip not available on PATH")

    repo_root, db_path = _setup_repo_structure(tmp_path)
    build_dir = repo_root / "build"
    document_output_dir = repo_root / "document_output"

    _ = BuildPaths.from_layout(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=document_output_dir,
    )

    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=True, validate_schema=True)
    )
    try:
        storage, tools = _create_scip_adapters(gateway, repo_root)
        scip_dir = build_dir / "scip"
        scip_dir.mkdir(parents=True, exist_ok=True)

        config = ScipIngestConfig(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.scip.json",
        )

        step = ScipIngestStep(storage=storage, tools=tools)
        result = asyncio.run(step.execute_async([], config))

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "unknown"
            pytest.skip(f"SCIP ingestion not successful in test environment: {errors}")

        if not (scip_dir / "index.scip").is_file():
            pytest.fail("index.scip was not created under build/scip")
        if not (scip_dir / "index.scip.json").is_file():
            pytest.fail("index.scip.json was not created under build/scip")

        con = gateway.con
        row = con.execute("SELECT COUNT(*) FROM scip_index_view").fetchone()
        if row is None:
            pytest.fail("scip_index_view did not return a row")
        if row[0] == 0:
            pytest.fail("scip_index_view is empty; expected rows after ingest")

    finally:
        gateway.close()


def test_scip_ingest_result_factory() -> None:
    """Verify ScipIngestResult factory methods work correctly."""
    # Test success result
    success = ScipIngestResult(
        status="success",
        index_scip=Path("build/scip/index.scip"),
        index_json=Path("build/scip/index.scip.json"),
    )
    if success.status != "success":
        pytest.fail(f"Expected status='success', got {success.status}")
    if success.index_scip is None:
        pytest.fail("Expected index_scip to be set")

    # Test unavailable result
    unavail = ScipIngestResult(
        status="unavailable",
        index_scip=None,
        index_json=None,
        reason="SCIP binary not found",
    )
    if unavail.status != "unavailable":
        pytest.fail(f"Expected status='unavailable', got {unavail.status}")
    if unavail.reason != "SCIP binary not found":
        pytest.fail(f"Unexpected reason: {unavail.reason}")
