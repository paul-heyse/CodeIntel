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
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.env import create_test_env
from tests._helpers.env_options import EnvOptions
from tests._helpers.fakes.contexts import (
    EnvOverrides,
    ExecutionContextBuilder,
    TargetResourceOverrides,
    make_test_output_target,
)
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
    type_checker: RecordingTypeChecker | None = None,
) -> TypingIngestPlugin:
    """Create a TypingIngestPlugin with recording adapters.

    Parameters
    ----------
    capture
        Capture object to record adapter and step calls.
    result
        Optional custom result to return from the step.
    type_checker
        Optional type checker instance.

    Returns
    -------
    TypingIngestPlugin
        Configured plugin instance.
    """
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


def _build_target_context(
    repo_root: Path,
    *,
    modules: tuple[str, ...] = (),
) -> TargetExecutionContext:
    """Build a TargetExecutionContext for typing plugin testing.

    Parameters
    ----------
    repo_root
        Repository root directory.
    modules
        Module paths to include in resources.

    Returns
    -------
    TargetExecutionContext
        Configured context for plugin execution.
    """
    env = create_test_env(repo_root, options=EnvOptions(repo_root=repo_root))
    builder = ExecutionContextBuilder.create(
        repo_root,
        env_overrides=EnvOverrides(
            gateway=env.gateway,
            snapshot=(DEFAULT_REPO, DEFAULT_COMMIT),
        ),
    )
    plugin = TypingIngestPlugin()
    target = make_test_output_target(plugin)

    # Note: type_checker is accessed through the plugin's type_checker_factory
    # rather than through resources. We include modules in resources.
    return builder.build_target_context(
        target,
        resources=TargetResourceOverrides(modules=modules),
    )


@pytest.mark.anyio
async def test_typing_plugin_skips_without_type_checker(tmp_path: Path) -> None:
    """When no type checker is provided, the plugin should skip work."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    ctx = _build_target_context(repo_root, modules=("pkg/typed.py",))

    result = await TypingIngestPlugin().execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})


@pytest.mark.anyio
async def test_typing_plugin_runs_step_and_returns_counts(tmp_path: Path) -> None:
    """Happy path: adapters are constructed and async step is executed."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    # Use RecordingTypeChecker (a proper double) instead of object()
    checker = RecordingTypeChecker()
    ctx = _build_target_context(repo_root, modules=("pkg/typed.py",))
    captured = StepCallCapture()

    # Pass the checker directly rather than via ctx.resources.type_checker
    result = await _make_plugin(captured, type_checker=checker).execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts, {"analytics.typedness": 1})
    expect_true(captured.storage is not None)
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    expect_true(getattr(captured.tool_port, "_type_checker", None) is checker)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/typed.py")
    expect_equal(module_record.file_path, repo_root / "pkg/typed.py")


@pytest.mark.anyio
async def test_typing_plugin_reports_failure(tmp_path: Path) -> None:
    """Errors returned from the async step should fail the target result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    # Use RecordingTypeChecker (a proper double) instead of object()
    checker = RecordingTypeChecker()
    ctx = _build_target_context(repo_root, modules=("pkg/typed.py",))
    captured = StepCallCapture()
    failing_result = StepResult.fail("typing blew up")

    # Pass the checker directly rather than via ctx.resources.type_checker
    result = await _make_plugin(
        captured,
        result=failing_result,
        type_checker=checker,
    ).execute(ctx)

    expect_true(result.success is False)
    expect_equal(result.error_message, "Typing ingest failed: typing blew up")
