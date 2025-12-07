"""Tests for TypingIngestPlugin behavior and error handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins import typing_plugin
from codeintel.ingestion.plugins.typing_plugin import TypingIngestPlugin
from codeintel.ingestion.ports.discovery import ModuleRecord
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context


@dataclass
class _Capture:
    gateway: object | None = None
    repo_root: Path | None = None
    modules: list[ModuleRecord] = field(default_factory=list)
    repo: str | None = None
    commit: str | None = None
    tools: object | None = None
    type_checker: object | None = None

TYPED_SOURCE = dedent(
    """\
    def fn(x: int) -> int:
        return x
    """
)


@pytest.mark.anyio
async def test_typing_plugin_skips_without_type_checker(tmp_path: Path) -> None:
    """When no type checker is provided, the plugin should skip work."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=None)

    result = await TypingIngestPlugin().execute(cast("TargetExecutionContext", ctx))

    assert result.success is True
    assert result.row_counts == {}


@pytest.mark.anyio
async def test_typing_plugin_runs_step_and_returns_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: adapters are constructed and async step is executed."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=object())
    captured = _Capture()

    class FakeStep:
        def __init__(self, *args: object, **kwargs: object) -> None:
            storage = kwargs.get("storage") or (args[0] if args else None)
            discovery = kwargs.get("discovery") or (args[1] if len(args) > 1 else None)
            tools = kwargs.get("tools") or (args[2] if len(args) > 2 else None)
            captured.gateway = storage
            captured.repo_root = cast("Path", discovery)
            captured.tools = tools

        async def execute_async(
            self,
            modules: list[ModuleRecord],
            *,
            repo: str,
            commit: str,
            repo_root: str,
        ) -> StepResult:
            captured.modules = modules
            captured.repo = repo
            captured.commit = commit
            captured.repo_root = Path(repo_root)
            return StepResult.ok(table_counts={"analytics.typedness": len(modules)})

    def fake_storage_adapter(gateway: object) -> object:
        captured.gateway = gateway
        return gateway

    def fake_discovery_adapter(repo_root_arg: Path) -> object:
        captured.repo_root = repo_root_arg
        return repo_root_arg

    def fake_build_tool_adapter(type_checker: object) -> object:
        captured.type_checker = type_checker
        return type_checker

    monkeypatch.setattr(typing_plugin, "DuckDBStorageAdapter", fake_storage_adapter)
    monkeypatch.setattr(typing_plugin, "FilesystemDiscoveryAdapter", fake_discovery_adapter)
    monkeypatch.setattr(typing_plugin, "BuildToolAdapter", fake_build_tool_adapter)
    monkeypatch.setattr(typing_plugin, "TypingIngestStep", FakeStep)

    result = await TypingIngestPlugin().execute(cast("TargetExecutionContext", ctx))

    assert result.success is True
    assert result.row_counts == {"analytics.typedness": 1}
    assert captured.gateway is ctx.gateway
    assert captured.type_checker is ctx.resources.type_checker
    assert captured.repo_root == repo_root
    assert captured.repo == DEFAULT_REPO
    assert captured.commit == DEFAULT_COMMIT
    module_record = captured.modules[0]
    assert module_record.rel_path == "pkg/typed.py"
    assert module_record.file_path == repo_root / "pkg/typed.py"


@pytest.mark.anyio
async def test_typing_plugin_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Errors returned from the async step should fail the target result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=object())

    class FakeStep:
        def __init__(self, *_: object, **__: object) -> None:
            return

        async def execute_async(self, *_: object, **__: object) -> StepResult:
            return StepResult.fail("typing blew up")

    monkeypatch.setattr(typing_plugin, "DuckDBStorageAdapter", lambda gateway: gateway)
    monkeypatch.setattr(typing_plugin, "FilesystemDiscoveryAdapter", lambda root: root)
    monkeypatch.setattr(typing_plugin, "BuildToolAdapter", lambda type_checker: type_checker)
    monkeypatch.setattr(typing_plugin, "TypingIngestStep", FakeStep)

    result = await TypingIngestPlugin().execute(cast("TargetExecutionContext", ctx))

    assert result.success is False
    assert result.error_message == "Typing ingest failed: typing blew up"
