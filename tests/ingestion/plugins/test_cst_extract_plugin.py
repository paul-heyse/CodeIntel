"""Tests for CstExtractPlugin wiring and fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins import cst_extract
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from codeintel.ingestion.ports.discovery import ModuleRecord
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.fakes.ingestion_context import RecordingGateway


@dataclass
class _Capture:
    gateway: object | None = None
    repo_root: Path | None = None
    modules: list[ModuleRecord] = field(default_factory=list)
    repo: str | None = None
    commit: str | None = None


@pytest.mark.anyio
async def test_execute_logs_errors_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Errors from the step should be logged but still return a success result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/cst_mod.py": "y = 2\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/cst_mod.py",))
    captured = _Capture()

    class FakeStep:
        def __init__(self, *args: object, **kwargs: object) -> None:
            storage = kwargs.get("storage") or (args[0] if args else None)
            discovery = kwargs.get("discovery") or (args[1] if len(args) > 1 else None)
            captured.gateway = storage
            captured.repo_root = cast("Path", discovery)
            self.modules: list[ModuleRecord] = []
            self.repo: str | None = None
            self.commit: str | None = None

        def execute(
            self,
            modules: list[ModuleRecord],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            self.modules = modules
            self.repo = repo
            self.commit = commit
            captured.modules = modules
            captured.repo = repo
            captured.commit = commit
            return StepResult(errors=["bad cst"])

    def fake_storage_adapter(gateway: object) -> object:
        captured.gateway = gateway
        return gateway

    def fake_discovery_adapter(repo_root_arg: Path) -> object:
        captured.repo_root = repo_root_arg
        return repo_root_arg

    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", fake_storage_adapter)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", fake_discovery_adapter)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    with caplog.at_level("WARNING", logger=cst_extract.log.name):
        result = await CstExtractPlugin().execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    expect_true(captured.gateway is ctx.gateway)
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/cst_mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/cst_mod.py")
    expect_true(any("bad cst" in record.getMessage() for record in caplog.records))


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway results should seed module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/from_db.py",)])
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = _Capture()

    class FakeStep:
        def __init__(self, *_: object, **__: object) -> None:
            self.modules: list[ModuleRecord] = []
            self.repo: str | None = None
            self.commit: str | None = None

        def execute(
            self,
            modules: list[ModuleRecord],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            self.modules = modules
            self.repo = repo
            self.commit = commit
            captured.modules = modules
            captured.repo = repo
            captured.commit = commit
            return StepResult.ok(table_counts={"core.cst_nodes": len(modules)})

    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    result = await CstExtractPlugin().execute(cast("TargetExecutionContext", ctx))

    expect_equal(result.row_counts, {"core.cst_nodes": 1})
    sql, params = gateway.executions[0]
    expect_in("core.modules", sql)
    expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/from_db.py")
    expect_equal(module_record.file_path, repo_root / "pkg/from_db.py")
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Database errors should be swallowed and an empty module list passed through."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway()
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured = _Capture()

    def _raise(*_: object, **__: object) -> object:
        message = "no db"
        raise OSError(message)

    class FakeStep:
        def __init__(self, *_: object, **__: object) -> None:
            self.modules: list[ModuleRecord] = []
            self.repo: str | None = None
            self.commit: str | None = None

        def execute(
            self,
            modules: list[ModuleRecord],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            self.modules = modules
            self.repo = repo
            self.commit = commit
            captured.modules = modules
            captured.repo = repo
            captured.commit = commit
            return StepResult.ok()

    monkeypatch.setattr(gateway.con, "execute", _raise)
    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    result = await CstExtractPlugin().execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(captured.modules, [])
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
