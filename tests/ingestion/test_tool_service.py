"""Tests for ToolService parsing and orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._helpers.tooling import ToolingOutputs, build_tooling_context, run_static_tooling


@pytest.fixture
def tooling_outputs(tmp_path: Path) -> ToolingOutputs:
    """
    Run the real tooling stack against a minimal repo.

    Returns
    -------
    ToolingOutputs
        Diagnostics and coverage reports produced by the tooling services.
    """
    context = build_tooling_context(tmp_path)
    return run_static_tooling(context)


def test_tool_service_pyright_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyright diagnostics per file."""
    errors = tooling_outputs.pyright_errors
    if errors.get("pkg/mod.py", 0) < 1:
        pytest.fail(f"Expected pyright to report errors for pkg/mod.py, got {errors}")


def test_tool_service_pyrefly_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyrefly diagnostics per file."""
    errors = tooling_outputs.pyrefly_errors
    if errors.get("pkg/mod.py", 0) < 1:
        pytest.fail(f"Expected pyrefly to report errors for pkg/mod.py, got {errors}")


def test_tool_service_coverage_reports(tooling_outputs: ToolingOutputs) -> None:
    """ToolService normalizes coverage.json payloads."""
    reports = {report.rel_path: report for report in tooling_outputs.coverage_reports}
    report = reports.get("pkg/mod.py")
    if report is None:
        pytest.fail(f"Coverage report missing for pkg/mod.py: {reports}")
    if not report.executed_lines:
        pytest.fail("Expected executed_lines to be populated for pkg/mod.py")
    if report.missing_lines:
        pytest.fail(f"Expected no missing lines, got {report.missing_lines}")
