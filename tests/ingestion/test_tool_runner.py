"""Tests for the ToolRunner abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import ToolName, ToolNotFoundError, ToolRunner


def test_tool_runner_missing_binary(tmp_path: Path) -> None:
    """Missing binary raises ToolNotFoundError."""
    runner = ToolRunner(
        cache_dir=tmp_path,
        tools_config=ToolsConfig(pyright_bin="does-not-exist"),
    )
    with pytest.raises(ToolNotFoundError):
        runner.run(ToolName.PYRIGHT, [], cwd=tmp_path)


def test_tool_runner_unknown_tool(tmp_path: Path) -> None:
    """Unknown tool names raise ValueError."""
    runner = ToolRunner(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown tool"):
        runner.run("unknown-tool", ["--version"])
