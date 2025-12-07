"""Tests for CstExtractPlugin wiring and fallbacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins import cst_extract
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.fakes.ingestion_context import RecordingGateway


@pytest.mark.anyio
async def test_execute_logs_errors_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Errors from the step should be logged but still return a success result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/cst_mod.py": "y = 2\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/cst_mod.py",))
    captured: dict[str, object] = {}

    class FakeStep:
        def __init__(self, storage: object, discovery: object) -> None:
            captured["storage"] = storage
            captured["discovery"] = discovery

        def execute(
            self,
            modules: list[object],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            captured["modules"] = modules
            captured["repo"] = repo
            captured["commit"] = commit
            return StepResult(errors=["bad cst"])

    def fake_storage_adapter(gateway: object) -> object:
        captured["gateway"] = gateway
        return object()

    def fake_discovery_adapter(repo_root_arg: Path) -> object:
        captured["repo_root"] = repo_root_arg
        return object()

    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", fake_storage_adapter)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", fake_discovery_adapter)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    with caplog.at_level("WARNING", logger=cst_extract.log.name):
        result = await CstExtractPlugin().execute(ctx)

    assert result.success is True
    assert result.row_counts == {}
    assert captured["gateway"] is ctx.gateway
    assert captured["repo_root"] == repo_root
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/cst_mod.py"
    assert module_record.file_path == repo_root / "pkg/cst_mod.py"
    assert any("bad cst" in record.getMessage() for record in caplog.records)


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway results should seed module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/from_db.py",)])
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured: dict[str, object] = {}

    class FakeStep:
        def __init__(self, *_: object) -> None:
            return

        def execute(
            self,
            modules: list[object],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            captured["modules"] = modules
            captured["repo"] = repo
            captured["commit"] = commit
            return StepResult.ok(table_counts={"core.cst_nodes": len(modules)})

    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    result = await CstExtractPlugin().execute(ctx)

    assert result.row_counts == {"core.cst_nodes": 1}
    sql, params = gateway.executions[0]
    assert "core.modules" in sql
    assert params == [DEFAULT_REPO, DEFAULT_COMMIT]
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/from_db.py"
    assert module_record.file_path == repo_root / "pkg/from_db.py"
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Database errors should be swallowed and an empty module list passed through."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway()
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured: dict[str, object] = {}

    def _raise(*_: object, **__: object) -> object:
        message = "no db"
        raise OSError(message)

    class FakeStep:
        def __init__(self, *_: object) -> None:
            return

        def execute(
            self,
            modules: list[object],
            *,
            repo: str,
            commit: str,
        ) -> StepResult:
            captured["modules"] = modules
            captured["repo"] = repo
            captured["commit"] = commit
            return StepResult.ok()

    monkeypatch.setattr(gateway.con, "execute", _raise)
    monkeypatch.setattr(cst_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(cst_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(cst_extract, "CstExtractStep", FakeStep)

    result = await CstExtractPlugin().execute(ctx)

    assert result.success is True
    assert captured["modules"] == []
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
