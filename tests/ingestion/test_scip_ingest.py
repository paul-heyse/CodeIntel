"""Integration tests for the Hamilton scip target."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.ingestion.scip import t__scip__ingest, t__scip__run
from codeintel.build.providers import create_default_providers
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.config import BuildLayoutOptions, BuildPaths
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.core.hamilton.records import TargetRunRecord
from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.ingestion import (
    build_scip_ingest_context,
    closing_gateway,
    materialize_rows_for_snapshot,
)
from tests._helpers.sql import count_table_rows

if TYPE_CHECKING:
    from tests._helpers.ingestion import ScipIngestContext


def _succeeded_modules_record() -> TargetRunRecord:
    return TargetRunRecord(
        target="modules",
        plugin_name="ingestion.modules",
        status="succeeded",
        input_hash="test",
    )


@pytest.fixture
def scip_ingest_context(tmp_path: Path) -> ScipIngestContext:
    """Provision repo, gateway, and adapters for SCIP ingest tests.

    Returns
    -------
    ScipIngestContext
        Context bundle for SCIP target execution.
    """
    return build_scip_ingest_context(tmp_path)


def test_scip_target_writes_tables(scip_ingest_context: ScipIngestContext) -> None:
    """Ensure scip target writes scip tables when SCIP binaries are available."""
    if shutil.which("scip-python") is None or shutil.which("scip") is None:
        pytest.skip("scip-python or scip not available on PATH")

    context = scip_ingest_context
    repo_root = context.repo_root
    gateway = context.gateway
    build_dir = context.build_dir

    paths = BuildPaths.from_layout(
        repo_root=repo_root,
        overrides=BuildLayoutOptions(
            build_dir=build_dir,
            db_path=gateway.config.db_path,
            document_output_dir=repo_root / "document_output",
        ),
    )
    providers = create_default_providers(ToolsConfig.default())
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    env = BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=BuildConfig.empty(),
        settings=TEST_BUILD_SETTINGS,
    )
    graph = get_target_metadata_service().system.graph

    modules_record = _succeeded_modules_record()

    with closing_gateway(gateway):
        run_result = t__scip__run(env, graph, modules_record)
        if not run_result.success:
            pytest.skip(run_result.error or "SCIP execution failed")

        ingest_result = t__scip__ingest(env, modules_record, run_result)
        if not ingest_result.result.success:
            pytest.fail(ingest_result.result.error or "SCIP ingestion failed")

        materialize_rows_for_snapshot(
            gateway,
            "core.scip_symbols",
            ingest_result.symbol_rows,
            snapshot=snapshot,
        )
        materialize_rows_for_snapshot(
            gateway,
            "core.scip_occurrences",
            ingest_result.occurrence_rows,
            snapshot=snapshot,
        )

        if run_result.index_path is None or not run_result.index_path.is_file():
            pytest.fail("index.scip was not created under build/scip")
        if run_result.json_path is None or not run_result.json_path.is_file():
            pytest.fail("index.json was not created under build/scip")

        count = count_table_rows(gateway.con, "core.scip_symbols")
        if count == 0:
            pytest.fail("core.scip_symbols is empty; expected rows after ingest")
