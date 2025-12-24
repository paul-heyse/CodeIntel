"""Integration checks for ToolRunner logging with real tool execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import ToolRunner, ToolRunOptions
from tests._helpers.assertions import expect_is_not_none, expect_true
from tests._helpers.tooling_audit import ToolCallLog, assert_tool_called


@pytest.mark.anyio
async def test_tool_runner_records_pytest_version(
    tmp_path: Path,
    tool_call_log: ToolCallLog,
) -> None:
    """Execute pytest and record the tool call log.

    Parameters
    ----------
    tmp_path
        Temporary filesystem root.
    tool_call_log
        Per-test tool call log recorder.
    """
    runner = ToolRunner(tools_config=ToolsConfig.default(), cache_dir=tmp_path)
    result = await runner.run_async(
        ToolName.PYTEST,
        ["--version"],
        options=ToolRunOptions(cwd=tmp_path),
    )

    expect_true(result.ok, message="Expected pytest to exit successfully.")
    call = assert_tool_called(
        tool_call_log.read(),
        ToolName.PYTEST,
        expected_args_contains=["--version"],
    )
    expect_is_not_none(call.version, message="Expected pytest version to be recorded.")
