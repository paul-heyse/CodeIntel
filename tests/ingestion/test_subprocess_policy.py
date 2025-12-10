"""Ensure subprocess usage is centralized in tool runner/service modules."""

from __future__ import annotations

import asyncio
from pathlib import Path

from codeintel.ingestion.engine.infrastructure import ToolName
from tests._helpers.assertions import (
    SUBPROCESS_ALLOWLIST,
    assert_no_subprocess_usage,
    expect_equal,
)
from tests._helpers.fakes.tools import PresetRunner, ToolRunOptions, make_tool_run_result


def test_no_direct_subprocess_usage_outside_tooling() -> None:
    """Fail when subprocess usage appears outside the centralized tooling modules."""
    repo_root = Path().resolve()
    src_root = repo_root / "src" / "codeintel"
    assert_no_subprocess_usage(src_root, allowed=SUBPROCESS_ALLOWLIST)


def test_preset_runner_respects_tool_run_options(tmp_path: Path) -> None:
    """PresetRunner should surface ToolRunOptions without subprocesses."""
    options = ToolRunOptions(
        returncode=1,
        stdout="out",
        stderr="err",
        duration_s=0.2,
    )
    preset_result = make_tool_run_result(ToolName.RUFF, options=options)
    runner = PresetRunner(preset_result)

    result = asyncio.run(runner.run_async(ToolName.RUFF, [], output_path=tmp_path / "out.json"))

    expect_equal(result.returncode, options.returncode)
    expect_equal(result.stdout, options.stdout)
    expect_equal(result.stderr, options.stderr)
    expect_equal(result.duration_s, options.duration_s)
