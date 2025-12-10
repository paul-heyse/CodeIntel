"""Tests for TypingIngestPlugin behavior and error handling."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.typing_plugin import StepFactory, TypingIngestPlugin
from tests._helpers.assertions import assert_logged, expect_equal, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.ingestion_plugins import (
    RecordingTypeChecker,
    StepCallCapture,
    make_recording_adapter_factories,
    make_recording_async_step_factory,
    make_recording_type_checker_factory,
)
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_repo_tree,
    run_ingestion_scenario,
)
from tests._helpers.ingestion_samples import TYPED_SOURCE


def _make_plugin(
    capture: StepCallCapture,
    *,
    result: StepResult | None = None,
    type_checker: RecordingTypeChecker | None = None,
) -> TypingIngestPlugin:
    checker = type_checker if type_checker is not None else RecordingTypeChecker()
    storage_factory, discovery_factory = make_recording_adapter_factories(capture)
    step_factory = cast(
        "StepFactory",
        make_recording_async_step_factory(capture, result=result),
    )
    return TypingIngestPlugin(
        storage_adapter_factory=storage_factory,
        discovery_adapter_factory=discovery_factory,
        type_checker_factory=make_recording_type_checker_factory(checker),
        step_factory=step_factory,
    )


@pytest.mark.anyio
async def test_typing_plugin_skips_without_type_checker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When no type checker is provided, the plugin should skip work."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/typed.py": TYPED_SOURCE})
    caplog.set_level("INFO")

    _ctx, result = await run_ingestion_scenario(
        TypingIngestPlugin,
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            resources=TargetResourceOverrides(modules=("pkg/typed.py",)),
        ),
    )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    assert_logged(caplog.records, level="INFO", containing="Type checker not available")


@pytest.mark.anyio
async def test_typing_plugin_runs_step_and_returns_counts(tmp_path: Path) -> None:
    """Happy path: adapters are constructed and async step is executed."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/typed.py": TYPED_SOURCE, "pkg/naive.py": TYPED_SOURCE},
    )
    # Use RecordingTypeChecker (a proper double) instead of object()
    checker = RecordingTypeChecker()
    captured = StepCallCapture()

    ctx, result = await run_ingestion_scenario(
        lambda: _make_plugin(captured, type_checker=checker),
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            resources=TargetResourceOverrides(modules=("pkg/typed.py", "pkg/naive.py")),
        ),
    )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {"analytics.typedness": 2})
    expect_true(captured.storage is not None)
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, ctx.repo)
    expect_equal(captured.commit, ctx.commit)
    expect_true(getattr(captured.tool_port, "_type_checker", None) is checker)
    recorded_paths = {record.rel_path for record in captured.modules}
    expect_equal(recorded_paths, {"pkg/typed.py", "pkg/naive.py"})


@pytest.mark.anyio
async def test_typing_plugin_reports_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Errors returned from the async step should fail the target result."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/typed.py": TYPED_SOURCE, "pkg/unicode/delta.py": TYPED_SOURCE},
    )
    # Use RecordingTypeChecker (a proper double) instead of object()
    checker = RecordingTypeChecker()
    captured = StepCallCapture()
    failing_result = StepResult.fail("typing blew up")
    caplog.set_level("WARNING")

    ctx, result = await run_ingestion_scenario(
        lambda: _make_plugin(captured, result=failing_result, type_checker=checker),
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            resources=TargetResourceOverrides(modules=("pkg/typed.py", "pkg/unicode/delta.py")),
        ),
    )

    expect_true(result.success is False)
    expect_equal(result.error_message, "Typing ingest failed: typing blew up")
    assert_logged(caplog.records, level="WARNING", containing="Typing ingest failed")
