"""Tests for DocstringsIngestPlugin and basic fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import DocstringsExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort
from codeintel.ingestion.ports.storage import IngestStoragePort
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
    result: StepResult | None = None,
    table_key: str = "core.docstrings",
) -> DocstringsIngestPlugin:
    return DocstringsIngestPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        step_factory=_build_step_factory(
            capture=capture,
            table_key=table_key,
            result=result,
        ),
    )


def _build_step_factory(
    *,
    capture: StepCallCapture,
    table_key: str,
    result: StepResult | None,
) -> Callable[[IngestStoragePort, ModuleDiscoveryPort], DocstringsExtractStep]:
    def _factory(
        storage: IngestStoragePort, discovery: ModuleDiscoveryPort
    ) -> DocstringsExtractStep:
        return cast(
            "DocstringsExtractStep",
            RecordingStep(
                storage,
                discovery,
                capture=capture,
                table_key=table_key,
                result=result,
            ),
        )

    return _factory


@pytest.mark.anyio
async def test_execute_persists_rows_and_logs_errors(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Successful result with logged errors should still return success."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/doc_mod.py": "def fn():\n    pass\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/doc_mod.py",))
    captured = StepCallCapture()
    failing_result = StepResult(errors=["warn"])

    with caplog.at_level("WARNING", logger="codeintel.ingestion.plugins.docstrings_plugin"):
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
    expect_equal(module_record.rel_path, "pkg/doc_mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/doc_mod.py")
    expect_true(any("warn" in record.getMessage() for record in caplog.records))


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(tmp_path: Path) -> None:
    """Gateway query should seed module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/db_doc.py",)])
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_equal(result.row_counts, {"core.docstrings": 1})
    sql, params = gateway.executions[0]
    expect_true("core.modules" in sql)
    expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/db_doc.py")
    expect_equal(module_record.file_path, repo_root / "pkg/db_doc.py")
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)


class _FailingConnection(_RecordingConnection):
    def __init__(self, gateway: RecordingGateway) -> None:
        super().__init__(gateway)

    @staticmethod
    def execute(sql: str, params: object) -> _RecordingConnection:
        _ = sql, params
        message = "db down"
        raise OSError(message)


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(tmp_path: Path) -> None:
    """Database errors should not cause failures and pass through empty modules."""
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
