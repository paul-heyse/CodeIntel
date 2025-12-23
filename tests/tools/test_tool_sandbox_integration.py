"""Integration checks for ToolSandbox with real ToolRunner execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import ToolRunner, ToolRunOptions
from tests._helpers.assertions import expect_true
from tests._helpers.tool_payloads import pytest_report_payload
from tests._helpers.tool_sandbox import ToolSandbox


@pytest.mark.anyio
async def test_tool_sandbox_executes_pytest(tmp_path: Path, tool_sandbox: ToolSandbox) -> None:
    """Execute a stubbed pytest binary via ToolRunner.

    Parameters
    ----------
    tmp_path
        Temporary filesystem root.
    tool_sandbox
        ToolSandbox with stubbed executables.
    """
    tool_sandbox.install_default_stubs()
    report_path = tmp_path / "pytest-report.json"
    runner = ToolRunner(tools_config=ToolsConfig.default(), cache_dir=tmp_path)

    with tool_sandbox.prepend_path():
        result = await runner.run_async(
            ToolName.PYTEST,
            ["--json-report", f"--json-report-file={report_path}"],
            options=ToolRunOptions(cwd=tmp_path, output_path=report_path),
        )

    expect_true(result.ok, message="Expected pytest stub to exit successfully.")
    expect_true(report_path.is_file(), message="Expected pytest report to be written.")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    expected = pytest_report_payload(tests=[], summary={"passed": 0, "failed": 0, "skipped": 0})
    expect_true(payload == expected, message="Expected pytest report payload to match defaults.")
