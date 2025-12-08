"""Tests for TypingIngestPlugin behavior and error handling."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import TypingIngestStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.typing_plugin import TypingIngestPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.ingestion_plugins import (
    RecordingAsyncStep,
    RecordingDiscoveryAdapter,
    RecordingStorageAdapter,
    RecordingTypeChecker,
    StepCallCapture,
    make_type_checker_factory,
)

TYPED_SOURCE = dedent(
    """\
    def fn(x: int) -> int:
        return x
    """
)


def _make_plugin(
    capture: StepCallCapture,
    *,
    result: StepResult | None = None,
    type_checker: object | None = None,
) -> TypingIngestPlugin:
    checker = type_checker if type_checker is not None else RecordingTypeChecker()
    return TypingIngestPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        type_checker_factory=make_type_checker_factory(checker),
        step_factory=lambda storage, discovery, tools: cast(
            "TypingIngestStep",
            RecordingAsyncStep(
                storage,
                discovery,
                tools,
                capture=capture,
                result=result,
            ),
        ),
    )


@pytest.mark.anyio
async def test_typing_plugin_skips_without_type_checker(tmp_path: Path) -> None:
    """When no type checker is provided, the plugin should skip work."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=None)

    result = await TypingIngestPlugin().execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})


@pytest.mark.anyio
async def test_typing_plugin_runs_step_and_returns_counts(tmp_path: Path) -> None:
    """Happy path: adapters are constructed and async step is executed."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=object())
    captured = StepCallCapture()

    result = await _make_plugin(captured, type_checker=ctx.resources.type_checker).execute(
        cast("TargetExecutionContext", ctx)
    )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {"analytics.typedness": 1})
    expect_true(captured.storage is not None)
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    expect_true(getattr(captured.tool_port, "_type_checker", None) is ctx.resources.type_checker)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/typed.py")
    expect_equal(module_record.file_path, repo_root / "pkg/typed.py")


@pytest.mark.anyio
async def test_typing_plugin_reports_failure(tmp_path: Path) -> None:
    """Errors returned from the async step should fail the target result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/typed.py",), type_checker=object())
    captured = StepCallCapture()
    failing_result = StepResult.fail("typing blew up")

    result = await _make_plugin(
        captured,
        result=failing_result,
        type_checker=ctx.resources.type_checker,
    ).execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is False)
    expect_equal(result.error_message, "Typing ingest failed: typing blew up")
