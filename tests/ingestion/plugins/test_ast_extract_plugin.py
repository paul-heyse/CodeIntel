"""Tests for AstExtractPlugin module wiring."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import AstExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.assertions import expect_equal, expect_true
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
    table_key: str = "core.ast_nodes",
    result: StepResult | None = None,
) -> AstExtractPlugin:
    return AstExtractPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        step_factory=lambda storage, discovery: cast(
            "AstExtractStep",
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
async def test_execute_invokes_step_and_returns_row_counts(tmp_path: Path) -> None:
    """Happy path: modules from resources flow through adapters to the step."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/mod.py",))
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(result.row_counts, {"core.ast_nodes": 1})
    expect_true(captured.storage is not None)
    expect_true(isinstance(captured.storage, RecordingStorageAdapter))
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/mod.py")


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(tmp_path: Path) -> None:
    """When modules are absent in resources, the gateway should be queried."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/db_mod.py",)])
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_equal(result.row_counts, {"core.ast_nodes": 1})
    sql, params = gateway.executions[0]
    expect_true("core.modules" in sql)
    expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/db_mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/db_mod.py")
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)


class _FailingConnection(_RecordingConnection):
    def __init__(self, gateway: RecordingGateway) -> None:
        super().__init__(gateway)

    def execute(self, sql: str, params: object) -> _RecordingConnection:
        _ = sql, params
        message = "db down"
        raise RuntimeError(message)


@pytest.mark.anyio
async def test_execute_recovers_from_gateway_errors(tmp_path: Path) -> None:
    """Database lookup failures should result in an empty module set."""
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
