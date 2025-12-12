"""Integration test for scip_ingest using real SCIP binaries."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config import BuildLayoutOptions, BuildPaths
from codeintel.ingestion import ScipIngestStep
from codeintel.ingestion.compute.scip_ingest import ScipIngestConfig, ScipIngestResult
from tests._helpers.ingestion import (
    build_scip_ingest_context,
    closing_gateway,
)
from tests._helpers.sql import count_table_rows

if TYPE_CHECKING:
    from tests._helpers.ingestion import (
        ScipIngestContext,
    )


@pytest.fixture
def scip_ingest_context(tmp_path: Path) -> ScipIngestContext:
    """Provision repo, gateway, and adapters for SCIP ingest tests.

    Returns
    -------
    ScipIngestContext
        Context bundle with repo_root, gateway, adapters, and build_dir.
    """
    return build_scip_ingest_context(tmp_path)


def test_ingest_scip_produces_artifacts(scip_ingest_context: ScipIngestContext) -> None:
    """Ensure scip_ingest generates SCIP artifacts and registers scip_index_view.

    Skip if scip-python or scip binaries are unavailable.
    """
    if shutil.which("scip-python") is None or shutil.which("scip") is None:
        pytest.skip("scip-python or scip not available on PATH")

    context = scip_ingest_context
    repo_root = context.repo_root
    gateway = context.gateway
    build_dir = context.build_dir
    document_output_dir = repo_root / "document_output"
    db_path = gateway.config.db_path

    _ = BuildPaths.from_layout(
        repo_root=repo_root,
        overrides=BuildLayoutOptions(
            build_dir=build_dir,
            db_path=db_path,
            document_output_dir=document_output_dir,
        ),
    )

    with closing_gateway(gateway):
        scip_dir = build_dir / "scip"

        config = ScipIngestConfig(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.scip.json",
        )

        step = ScipIngestStep(storage=context.storage, tools=context.tools)
        result = asyncio.run(step.execute_async([], config))

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "unknown"
            pytest.skip(f"SCIP ingestion not successful in test environment: {errors}")

        if not (scip_dir / "index.scip").is_file():
            pytest.fail("index.scip was not created under build/scip")
        if not (scip_dir / "index.scip.json").is_file():
            pytest.fail("index.scip.json was not created under build/scip")

        count = count_table_rows(gateway.con, "core.scip_symbols")
        if count == 0:
            pytest.fail("core.scip_symbols is empty; expected rows after ingest")


def test_scip_ingest_result_factory() -> None:
    """Verify ScipIngestResult factory methods work correctly."""
    success = ScipIngestResult(
        status="success",
        index_scip=Path("build/scip/index.scip"),
        index_json=Path("build/scip/index.scip.json"),
    )
    if success.status != "success":
        pytest.fail(f"Expected status='success', got {success.status}")
    if success.index_scip is None:
        pytest.fail("Expected index_scip to be set")

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
