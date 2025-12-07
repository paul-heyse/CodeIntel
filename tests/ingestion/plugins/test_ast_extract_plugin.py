"""Tests for AstExtractPlugin module wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins import ast_extract
from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.fakes.ingestion_context import RecordingGateway


@pytest.mark.anyio
async def test_execute_invokes_step_and_returns_row_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: modules from resources flow through adapters to the step."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/mod.py",))
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
            return StepResult.ok(table_counts={"core.ast_nodes": len(modules)})

    def fake_storage_adapter(gateway: object) -> object:
        captured["gateway"] = gateway
        return object()

    def fake_discovery_adapter(repo_root_arg: Path) -> object:
        captured["repo_root"] = repo_root_arg
        return object()

    monkeypatch.setattr(ast_extract, "DuckDBStorageAdapter", fake_storage_adapter)
    monkeypatch.setattr(ast_extract, "FilesystemDiscoveryAdapter", fake_discovery_adapter)
    monkeypatch.setattr(ast_extract, "AstExtractStep", FakeStep)

    result = await AstExtractPlugin().execute(ctx)

    assert result.success is True
    assert result.row_counts == {"core.ast_nodes": 1}
    assert captured["gateway"] is ctx.gateway
    assert captured["repo_root"] == repo_root
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/mod.py"
    assert module_record.file_path == repo_root / "pkg/mod.py"


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When modules are absent in resources, the gateway should be queried."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/db_mod.py",)])
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
            return StepResult.ok(table_counts={"core.ast_nodes": len(modules)})

    monkeypatch.setattr(ast_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(ast_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(ast_extract, "AstExtractStep", FakeStep)

    result = await AstExtractPlugin().execute(ctx)

    assert result.row_counts == {"core.ast_nodes": 1}
    sql, params = gateway.executions[0]
    assert "core.modules" in sql
    assert params == [DEFAULT_REPO, DEFAULT_COMMIT]
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/db_mod.py"
    assert module_record.file_path == repo_root / "pkg/db_mod.py"
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT


@pytest.mark.anyio
async def test_execute_recovers_from_gateway_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Database lookup failures should result in an empty module set."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway()
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured: dict[str, object] = {}

    def _raise(*_: object, **__: object) -> object:
        message = "db down"
        raise RuntimeError(message)

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
    monkeypatch.setattr(ast_extract, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(ast_extract, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(ast_extract, "AstExtractStep", FakeStep)

    result = await AstExtractPlugin().execute(ctx)

    assert result.success is True
    assert captured["modules"] == []
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
