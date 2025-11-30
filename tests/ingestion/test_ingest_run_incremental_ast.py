"""Verify incremental view metrics are attached to AST ingest runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.ingest_runs import IngestRun, IngestRunSink
from codeintel.ingestion.runner import IngestionContext, run_ast_extract, run_repo_scan
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
from tests._helpers.gateway import open_ingestion_gateway


@dataclass
class RecordingSink(IngestRunSink):
    """Simple sink for capturing run records in-memory."""

    runs: list[IngestRun]

    def record(self, run: IngestRun) -> None:
        """Store the provided run for later assertions."""
        self.runs.append(run)


def test_ast_extract_ingest_run_includes_incremental_view_metrics(tmp_path: Path) -> None:
    """Ensure incremental view metrics are propagated into IngestRun."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")
    (src_dir / "b.py").write_text("print('b')\n", encoding="utf8")

    snapshot = SnapshotRef.from_args(repo="demo/ast", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root)
    ctx = IngestionContext(
        snapshot=snapshot,
        paths=paths,
        gateway=open_ingestion_gateway(),
        tools=ToolsConfig.default(),
        code_profile_cfg=default_code_profile(repo_root),
        config_profile_cfg=default_config_profile(repo_root),
    )
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = False

    run_repo_scan(ctx)
    run_ast_extract(ctx)

    minimum_modules = 2
    ast_runs = [run for run in sink.runs if run.step == "ast_extract"]
    if not ast_runs:
        pytest.fail("expected at least one ast_extract IngestRun")
    run = ast_runs[0]

    if run.modules_total is None or run.modules_total < minimum_modules:
        pytest.fail(f"modules_total should be populated, got {run.modules_total}")
    if run.modules_changed is None:
        pytest.fail("modules_changed should be populated")
    if run.modules_deleted is None:
        pytest.fail("modules_deleted should be populated")
    if run.modules_changed_ratio is None:
        pytest.fail("modules_changed_ratio should be populated")
    if run.modules_deleted_ratio is None:
        pytest.fail("modules_deleted_ratio should be populated")
    if run.use_full_rebuild not in {True, False}:
        pytest.fail(f"use_full_rebuild should be a boolean, got {run.use_full_rebuild}")
