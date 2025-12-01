"""Tests for ingestion run reporting and error classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.ingest_runs import IngestRun, IngestRunSink, IngestRunStatus
from codeintel.ingestion.runner import (
    IngestionContext,
    run_docstrings_ingest,
    run_repo_scan,
)
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
from tests._helpers.gateway import open_ingestion_gateway


@dataclass
class RecordingSink(IngestRunSink):
    """Simple sink for capturing run records in-memory."""

    runs: list[IngestRun]

    def record(self, run: IngestRun) -> None:
        """Store the provided run for later inspection."""
        self.runs.append(run)


def _build_context(tmp_path: Path) -> IngestionContext:
    repo_root = tmp_path / "repo"
    paths = BuildPaths.from_repo_root(repo_root)
    snapshot = SnapshotRef.from_args(repo="demo/repo", commit="abc123", repo_root=repo_root)
    gateway = open_ingestion_gateway()

    return IngestionContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=ToolsConfig.default(),
        code_profile_cfg=default_code_profile(repo_root),
        config_profile_cfg=default_config_profile(repo_root),
    )


def test_ingest_run_success_reporting(tmp_path: Path) -> None:
    """Ensure runs are recorded for successful steps with metrics."""
    repo_root = tmp_path / "repo" / "src" / "pkg"
    repo_root.mkdir(parents=True)
    (repo_root / "a.py").write_text('"""docstring"""\n', encoding="utf8")

    ctx = _build_context(tmp_path)
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = True

    expected_runs = 2

    run_repo_scan(ctx)
    run_docstrings_ingest(ctx)

    if len(sink.runs) != expected_runs:
        pytest.fail(f"Expected {expected_runs} ingest runs, got {len(sink.runs)}")
    step_names = {run.step for run in sink.runs}
    if step_names != {"repo_scan", "docstrings_ingest"}:
        pytest.fail(f"Unexpected step names recorded: {step_names}")

    try:
        doc_run = next(run for run in sink.runs if run.step == "docstrings_ingest")
    except StopIteration:
        pytest.fail("docstrings_ingest run not recorded")  # pragma: no cover
    if doc_run.status not in {IngestRunStatus.OK, IngestRunStatus.SKIPPED}:
        pytest.fail(f"Unexpected status for docstrings_ingest: {doc_run.status}")
    if "core.docstrings" not in doc_run.datasets:
        pytest.fail(f"core.docstrings missing from datasets: {doc_run.datasets}")
    if doc_run.rows_inserted < 0:
        pytest.fail(f"rows_inserted should be non-negative, got {doc_run.rows_inserted}")
    if doc_run.duration_s is None:
        pytest.fail("duration_s was not populated")
    if not doc_run.run_id:
        pytest.fail("run_id was not set")


def test_ingest_run_error_classification(tmp_path: Path) -> None:
    """Force an error in docstrings_ingest and assert error_kind is set."""
    repo_root = tmp_path / "repo" / "src" / "pkg"
    repo_root.mkdir(parents=True)
    (repo_root / "a.py").write_text("not:python:code\n", encoding="utf8")

    ctx = _build_context(tmp_path)
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = False

    run_repo_scan(ctx)

    message = "synthetic parse error"

    def _boom(_ctx: IngestionContext) -> None:
        """
        Raise a synthetic parse error to test classification.

        Raises
        ------
        ValueError
            Always raised to emulate a parsing failure.
        """
        raise ValueError(message)

    ctx.step_overrides = {"docstrings_ingest": _boom}

    with pytest.raises(ValueError, match=message):
        run_docstrings_ingest(ctx)

    error_runs = [run for run in sink.runs if run.step == "docstrings_ingest"]
    if not error_runs:
        pytest.fail("Expected an error IngestRun to be recorded")
    run = error_runs[0]
    if run.status is not IngestRunStatus.ERROR:
        pytest.fail(f"Unexpected status for error run: {run.status}")
    if run.error_kind not in {"parse_error", "ValueError"}:
        pytest.fail(f"Unexpected error_kind: {run.error_kind}")
    if run.error_message is None or message not in run.error_message:
        pytest.fail(f"Unexpected error_message: {run.error_message}")
