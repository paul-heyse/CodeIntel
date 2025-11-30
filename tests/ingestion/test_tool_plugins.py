"""Test ingestion tool plugins."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.tools import ToolStatus, build_default_registry
from codeintel.ingestion.tools.pyright import PyrightPlugin


class DummyRunner(ToolRunner):
    """Provide a test double that bypasses real subprocess execution."""

    def __init__(self, result: ToolRunResult | Exception) -> None:
        self._result = result
        super().__init__(tools_config=ToolsConfig.default(), cache_dir=Path("build/.tool_cache"))

    async def run_async(  # type: ignore[override]
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """
        Return a preset ToolRunResult or raise the configured exception.

        Returns
        -------
        ToolRunResult
            Pre-baked result configured for the dummy runner.

        Raises
        ------
        ToolExecutionError
            Raised when the dummy runner is configured with a generic exception.
        ToolNotFoundError
            Raised when the dummy runner is configured with ToolNotFoundError.
        """
        del tool, args, cwd, timeout_s
        if isinstance(self._result, ToolNotFoundError):
            raise ToolNotFoundError(self._result.tool, self._result.configured_path)
        if isinstance(self._result, Exception):
            raise ToolExecutionError(
                ToolRunResult(
                    tool=ToolName.PYRIGHT,
                    args=(),
                    returncode=1,
                    stdout="",
                    stderr="dummy error",
                    duration_s=0.1,
                    output_path=output_path,
                )
            ) from self._result
        return self._result


def test_pyright_plugin_not_found_downgrades() -> None:
    """Ensure PyrightPlugin reports NOT_FOUND when the binary is missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYRIGHT, tools_cfg.pyright_bin)
    runner = DummyRunner(exc)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path()))

    if result.status is not ToolStatus.NOT_FOUND:
        pytest.fail(f"Expected NOT_FOUND status, got {result.status}")
    if result.run is not None:
        pytest.fail(f"Expected run to be None, got {result.run}")
    if not isinstance(result.error, ToolNotFoundError):
        pytest.fail(f"Expected ToolNotFoundError, got {result.error!r}")


def test_pyright_plugin_ok_status() -> None:
    """Confirm PyrightPlugin preserves successful ToolRunResult."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("--outputjson", "."),
        returncode=0,
        stdout='{"summary": {"files": {}}}',
        stderr="",
        duration_s=0.01,
        output_path=None,
    )
    runner = DummyRunner(run)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path()))

    if result.status is not ToolStatus.OK:
        pytest.fail(f"Expected OK status, got {result.status}")
    if not result.ok:
        pytest.fail("Expected result.ok to be True")
    if result.run is not run:
        pytest.fail(f"Expected run to be {run}, got {result.run}")
    if result.error is not None:
        pytest.fail(f"Expected no error, got {result.error!r}")


def test_default_registry_contains_plugins() -> None:
    """Verify the registry builder wires all expected plugin names."""
    runner = DummyRunner(
        ToolRunResult(
            tool=ToolName.PYRIGHT,
            args=(),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )
    )
    registry = build_default_registry(runner, runner.tools_config)
    names = registry.names()

    for plugin_name in ("pyright", "pyrefly", "ruff", "coverage", "pytest", "scip"):
        if plugin_name not in names:
            pytest.fail(f"Expected plugin {plugin_name} in registry, got {names}")
