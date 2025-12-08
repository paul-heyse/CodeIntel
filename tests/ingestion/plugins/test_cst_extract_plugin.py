"""Tests for CstExtractPlugin wiring and fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import CstExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.fakes.ingestion_context import RecordingGateway, _RecordingConnection
from tests._helpers.fakes.ingestion_plugins import (
    RecordingDiscoveryAdapter,
    RecordingStep,
    RecordingStorageAdapter,
    StepCallCapture,
)


def _make_plugin(
    capture: StepCallCapture,
    *,
    result: StepResult | None = None,
    table_key: str = "core.cst_nodes",
) -> CstExtractPlugin:
    return CstExtractPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        step_factory=lambda storage, discovery: cast(
            "CstExtractStep",
            RecordingStep(
                storage,
                discovery,
                capture=capture,
                table_key=table_key,
                result=result,
            ),
        ),
    )


@pytest.mark.anyio
async def test_execute_logs_errors_and_succeeds(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Errors from the step should be logged but still return a success result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/cst_mod.py": "y = 2\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/cst_mod.py",))
    captured = StepCallCapture()
    failing_result = StepResult(errors=["bad cst"])

    with caplog.at_level("WARNING", logger="codeintel.ingestion.plugins.cst_extract"):
        result = await _make_plugin(captured, result=failing_result).execute(
            cast("TargetExecutionContext", ctx)
        )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    expect_true(isinstance(captured.storage, RecordingStorageAdapter))
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/cst_mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/cst_mod.py")
    expect_true(any("bad cst" in record.getMessage() for record in caplog.records))


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(tmp_path: Path) -> None:
    """Gateway results should seed module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/from_db.py",)])
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_equal(result.row_counts, {"core.cst_nodes": 1})
    sql, params = gateway.executions[0]
    expect_in("core.modules", sql)
    expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/from_db.py")
    expect_equal(module_record.file_path, repo_root / "pkg/from_db.py")
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)


class _FailingConnection(_RecordingConnection):
    def __init__(self, gateway: RecordingGateway) -> None:
        super().__init__(gateway)

    def execute(self, sql: str, params: object) -> _RecordingConnection:
        _ = sql, params
        message = "no db"
        raise OSError(message)


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(tmp_path: Path) -> None:
    """Database errors should be swallowed and an empty module list passed through."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway()
    gateway.con = _FailingConnection(gateway)
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(captured.modules, [])
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
