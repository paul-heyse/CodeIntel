"""Tests for DocstringsIngestPlugin wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins import docstrings_plugin
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.fakes.ingestion_context import RecordingGateway


@pytest.mark.anyio
async def test_execute_passes_modules_and_returns_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docstring step should receive module records and propagate counts."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/doc_mod.py": '"""Doc."""\n'})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/doc_mod.py",))
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
            return StepResult.ok(table_counts={"core.docstrings": 5})

    def fake_storage_adapter(gateway: object) -> object:
        captured["gateway"] = gateway
        return object()

    def fake_discovery_adapter(repo_root_arg: Path) -> object:
        captured["repo_root"] = repo_root_arg
        return object()

    monkeypatch.setattr(docstrings_plugin, "DuckDBStorageAdapter", fake_storage_adapter)
    monkeypatch.setattr(docstrings_plugin, "FilesystemDiscoveryAdapter", fake_discovery_adapter)
    monkeypatch.setattr(docstrings_plugin, "DocstringsExtractStep", FakeStep)

    result = await DocstringsIngestPlugin().execute(ctx)

    assert result.success is True
    assert result.row_counts == {"core.docstrings": 5}
    assert captured["gateway"] is ctx.gateway
    assert captured["repo_root"] == repo_root
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/doc_mod.py"
    assert module_record.file_path == repo_root / "pkg/doc_mod.py"


@pytest.mark.anyio
async def test_execute_uses_gateway_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway rows should populate module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway(result_rows=[("pkg/doc_db.py",)])
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
            return StepResult.ok(table_counts={"core.docstrings": len(modules)})

    monkeypatch.setattr(docstrings_plugin, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(docstrings_plugin, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(docstrings_plugin, "DocstringsExtractStep", FakeStep)

    result = await DocstringsIngestPlugin().execute(ctx)

    assert result.row_counts == {"core.docstrings": 1}
    sql, params = gateway.executions[0]
    assert "core.modules" in sql
    assert params == [DEFAULT_REPO, DEFAULT_COMMIT]
    module_record = captured["modules"][0]
    assert module_record.rel_path == "pkg/doc_db.py"
    assert module_record.file_path == repo_root / "pkg/doc_db.py"
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway errors should yield an empty module list while succeeding."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    gateway = RecordingGateway()
    ctx = make_target_context(repo_root=repo_root, modules=(), gateway=gateway)
    captured: dict[str, object] = {}

    def _raise(*_: object, **__: object) -> object:
        message = "db error"
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
    monkeypatch.setattr(docstrings_plugin, "DuckDBStorageAdapter", lambda gateway_arg: gateway_arg)
    monkeypatch.setattr(docstrings_plugin, "FilesystemDiscoveryAdapter", lambda root_arg: root_arg)
    monkeypatch.setattr(docstrings_plugin, "DocstringsExtractStep", FakeStep)

    result = await DocstringsIngestPlugin().execute(ctx)

    assert result.success is True
    assert captured["modules"] == []
    assert captured["repo"] == DEFAULT_REPO
    assert captured["commit"] == DEFAULT_COMMIT
