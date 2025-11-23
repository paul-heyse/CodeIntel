"""Tests for the ToolRunner abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.tool_runner import ToolResult, ToolRunner

MISSING = 127


def test_tool_runner_missing_binary(tmp_path: Path) -> None:
    """Missing binary yields 127 and structured result."""
    runner = ToolRunner(cache_dir=tmp_path)
    result = runner.run("pyright", ["nonexistent-binary"], cwd=tmp_path)
    if not isinstance(result, ToolResult):
        pytest.fail("ToolRunner did not return ToolResult")
    if result.returncode != MISSING:
        pytest.fail(f"Expected returncode {MISSING}, got {result.returncode}")
    if "not found" not in result.stderr:
        pytest.fail("Expected missing binary message in stderr")


def test_tool_runner_empty_args(tmp_path: Path) -> None:
    """Empty args raise ValueError."""
    runner = ToolRunner(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="args must include"):
        runner.run("pyright", [], cwd=tmp_path)
