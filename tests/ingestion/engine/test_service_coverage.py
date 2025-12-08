"""Coverage tests for engine service module.

This module provides comprehensive tests for error paths in ToolService,
covering the uncovered lines in service.py including error handling
for run_pyright, run_pyrefly, run_ruff, run_coverage, run_scip, and run_pytest.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import override

import pytest

from codeintel.ingestion.engine import ToolPluginResult
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.engine.results import CoverageReport
from codeintel.ingestion.engine.service import ToolService
from tests._helpers.assertions import expect_equal, expect_is_instance, expect_true
from tests._helpers.fakes.tools import ToolRunOptions, make_tool_run_result

# =============================================================================
# Test Runner Implementations
# =============================================================================


class SuccessRunner(ToolRunner):
    """Runner that returns successful results with diagnostic output."""

    def __init__(self, cache_dir: Path, diagnostics: dict[str, int] | None = None) -> None:
        """Initialize with cache dir and optional diagnostics.

        Parameters
        ----------
        cache_dir
            Cache directory for tool outputs.
        diagnostics
            Optional mapping of paths to error counts.
        """
        super().__init__(cache_dir=cache_dir)
        self._diagnostics = diagnostics or {}
        self.calls: list[tuple[ToolName, tuple[str, ...]]] = []

    @override
    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """Return a successful result.

        Returns
        -------
        ToolRunResult
            Successful run result with diagnostic output.
        """
        _ = cwd, timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        self.calls.append((tool_enum, tuple(args)))

        # Generate diagnostic output if diagnostics configured
        stdout = ""
        if self._diagnostics:
            lines = [
                f'  "file": "{path}", "errors": {count}'
                for path, count in self._diagnostics.items()
            ]
            stdout = "{\n" + ",\n".join(lines) + "\n}"

        return make_tool_run_result(
            tool_enum,
            args=args,
            options=ToolRunOptions(
                returncode=0,
                stdout=stdout,
                stderr="",
                output_path=output_path,
                duration_s=0.1,
            ),
        )


class FailingRunner(ToolRunner):
    """Runner that returns failure status."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        return_code: int = 1,
        raise_exception: bool = False,
    ) -> None:
        """Initialize with cache dir and failure configuration.

        Parameters
        ----------
        cache_dir
            Cache directory for tool outputs.
        return_code
            Return code to use (non-zero for failure).
        raise_exception
            Whether to raise an exception instead of returning result.
        """
        super().__init__(cache_dir=cache_dir)
        self._return_code = return_code
        self._raise_exception = raise_exception
        self.calls: list[tuple[ToolName, tuple[str, ...]]] = []

    @override
    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """Return a failure result or raise an exception.

        Raises
        ------
        ToolExecutionError
            If configured to raise exception.

        Returns
        -------
        ToolRunResult
            Failed run result.
        """
        _ = cwd, timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        self.calls.append((tool_enum, tuple(args)))

        if self._raise_exception:
            result = make_tool_run_result(
                tool_enum,
                args=args,
                options=ToolRunOptions(
                    returncode=self._return_code,
                    stderr="Tool execution failed",
                    output_path=output_path,
                    duration_s=0.1,
                ),
            )
            raise ToolExecutionError(result)

        return make_tool_run_result(
            tool_enum,
            args=args,
            options=ToolRunOptions(
                returncode=self._return_code,
                stderr="Command failed",
                output_path=output_path,
                duration_s=0.1,
            ),
        )


class NotFoundRunner(ToolRunner):
    """Runner that simulates binary not found."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize with cache dir.

        Parameters
        ----------
        cache_dir
            Cache directory for tool outputs.
        """
        super().__init__(cache_dir=cache_dir)
        self.calls: list[tuple[ToolName, tuple[str, ...]]] = []

    @override
    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """Raise ToolNotFoundError to simulate missing binary.

        Raises
        ------
        ToolNotFoundError
            Always raised to simulate binary not found.
        """
        _ = cwd, output_path, timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        self.calls.append((tool_enum, tuple(args)))
        raise ToolNotFoundError(tool_enum, "/nonexistent/path")


# =============================================================================
# ToolService run_pyright Tests
# =============================================================================


class TestToolServiceRunPyright:
    """Tests for ToolService.run_pyright error paths."""

    @staticmethod
    def test_pyright_not_found_returns_empty(tmp_path: Path) -> None:
        """run_pyright returns empty dict when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        result = asyncio.run(service.run_pyright(tmp_path))

        expect_equal(result, {})

    @staticmethod
    def test_pyright_execution_failure_raises(tmp_path: Path) -> None:
        """run_pyright raises ToolExecutionError on failure."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=2, raise_exception=True)
        service = ToolService(runner)

        with pytest.raises(ToolExecutionError):
            asyncio.run(service.run_pyright(tmp_path))


# =============================================================================
# ToolService run_pyrefly Tests
# =============================================================================


class TestToolServiceRunPyrefly:
    """Tests for ToolService.run_pyrefly error paths."""

    @staticmethod
    def test_pyrefly_not_found_returns_empty(tmp_path: Path) -> None:
        """run_pyrefly returns empty dict when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        result = asyncio.run(service.run_pyrefly(tmp_path))

        expect_equal(result, {})

    @staticmethod
    def test_pyrefly_failure_returns_empty(tmp_path: Path) -> None:
        """run_pyrefly returns empty dict on failure (graceful degradation)."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=1)
        service = ToolService(runner)

        result = asyncio.run(service.run_pyrefly(tmp_path))

        # Pyrefly failures are handled gracefully, returning empty dict
        expect_equal(result, {})


# =============================================================================
# ToolService run_ruff Tests
# =============================================================================


class TestToolServiceRunRuff:
    """Tests for ToolService.run_ruff error paths."""

    @staticmethod
    def test_ruff_not_found_returns_empty(tmp_path: Path) -> None:
        """run_ruff returns empty dict when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        result = asyncio.run(service.run_ruff(tmp_path))

        expect_equal(result, {})

    @staticmethod
    def test_ruff_failure_raises(tmp_path: Path) -> None:
        """run_ruff raises ToolExecutionError on failure."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=2, raise_exception=True)
        service = ToolService(runner)

        with pytest.raises(ToolExecutionError):
            asyncio.run(service.run_ruff(tmp_path))


# =============================================================================
# ToolService run_coverage_report Tests
# =============================================================================


class TestToolServiceRunCoverageReport:
    """Tests for ToolService.run_coverage_report error paths."""

    @staticmethod
    def test_coverage_not_found_returns_empty(tmp_path: Path) -> None:
        """run_coverage_report returns empty report when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        result = asyncio.run(
            service.run_coverage_report(tmp_path, output_path=tmp_path / "cov.json")
        )

        # Coverage not found returns empty report (graceful degradation)
        expect_equal(result, CoverageReport.empty())

    @staticmethod
    def test_coverage_failure_returns_empty(tmp_path: Path) -> None:
        """run_coverage_report returns empty report on failure (graceful degradation)."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=1)
        service = ToolService(runner)

        result = asyncio.run(
            service.run_coverage_report(tmp_path, output_path=tmp_path / "cov.json")
        )

        # Coverage failures are handled gracefully
        expect_equal(result, CoverageReport.empty())


# =============================================================================
# ToolService run_scip_full Tests
# =============================================================================


class TestToolServiceRunScipFull:
    """Tests for ToolService.run_scip_full error paths."""

    @staticmethod
    def test_scip_not_found_raises(tmp_path: Path) -> None:
        """run_scip_full raises when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        with pytest.raises(ToolNotFoundError):
            asyncio.run(
                service.run_scip_full(
                    repo_root=tmp_path,
                    output_scip=tmp_path / "index.scip",
                    output_json=tmp_path / "index.json",
                )
            )

    @staticmethod
    def test_scip_execution_error_raises(tmp_path: Path) -> None:
        """run_scip_full raises ToolExecutionError on failure."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=1, raise_exception=True)
        service = ToolService(runner)

        with pytest.raises(ToolExecutionError):
            asyncio.run(
                service.run_scip_full(
                    repo_root=tmp_path,
                    output_scip=tmp_path / "index.scip",
                    output_json=tmp_path / "index.json",
                )
            )


# =============================================================================
# ToolService run_pytest_report Tests
# =============================================================================


class TestToolServiceRunPytestReport:
    """Tests for ToolService.run_pytest_report error paths."""

    @staticmethod
    def test_pytest_not_found_raises(tmp_path: Path) -> None:
        """run_pytest_report raises when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        # pytest not found should raise ToolNotFoundError
        with pytest.raises(ToolNotFoundError):
            asyncio.run(
                service.run_pytest_report(
                    repo_root=tmp_path,
                    json_report_path=tmp_path / "report.json",
                )
            )

    @staticmethod
    def test_pytest_execution_error_raises(tmp_path: Path) -> None:
        """run_pytest_report raises ToolExecutionError on failure."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=2, raise_exception=True)
        service = ToolService(runner)

        # pytest error (not test failures) should raise
        with pytest.raises(ToolExecutionError):
            asyncio.run(
                service.run_pytest_report(
                    repo_root=tmp_path,
                    json_report_path=tmp_path / "report.json",
                )
            )


# =============================================================================
# ToolService run_scip_shard Tests
# =============================================================================


class TestToolServiceRunScipShard:
    """Tests for ToolService.run_scip_shard error paths."""

    @staticmethod
    def test_scip_shard_not_found_raises(tmp_path: Path) -> None:
        """run_scip_shard raises when binary not found."""
        runner = NotFoundRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        with pytest.raises(ToolNotFoundError):
            asyncio.run(
                service.run_scip_shard(
                    repo_root=tmp_path,
                    rel_paths=["module.py"],
                    output_scip=tmp_path / "index.scip",
                    output_json=tmp_path / "index.json",
                )
            )

    @staticmethod
    def test_scip_shard_execution_error_raises(tmp_path: Path) -> None:
        """run_scip_shard raises ToolExecutionError on failure."""
        runner = FailingRunner(cache_dir=tmp_path, return_code=1, raise_exception=True)
        service = ToolService(runner)

        with pytest.raises(ToolExecutionError):
            asyncio.run(
                service.run_scip_shard(
                    repo_root=tmp_path,
                    rel_paths=["module.py"],
                    output_scip=tmp_path / "index.scip",
                    output_json=tmp_path / "index.json",
                )
            )


# =============================================================================
# ToolService get_plugin Tests
# =============================================================================


class TestToolServiceGetPlugin:
    """Tests for ToolService.get_plugin."""

    @staticmethod
    def test_get_plugin_returns_registered(tmp_path: Path) -> None:
        """get_plugin returns a registered plugin."""
        runner = SuccessRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        # "pyright" should be a registered plugin
        plugin = service.get_plugin("pyright")
        expect_true(plugin is not None)

    @staticmethod
    def test_get_plugin_unknown_raises(tmp_path: Path) -> None:
        """get_plugin raises KeyError for unknown plugin."""
        runner = SuccessRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        with pytest.raises(KeyError):
            service.get_plugin("unknown_plugin")


# =============================================================================
# ToolService run_plugin Tests
# =============================================================================


class TestToolServiceRunPlugin:
    """Tests for ToolService.run_plugin."""

    @staticmethod
    def test_run_plugin_unknown_raises(tmp_path: Path) -> None:
        """run_plugin raises KeyError for unknown plugin name."""
        runner = SuccessRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        with pytest.raises(KeyError, match="unknown_plugin"):
            asyncio.run(service.run_plugin("unknown_plugin", repo_root=tmp_path))

    @staticmethod
    def test_run_plugin_success_returns_result(tmp_path: Path) -> None:
        """run_plugin returns ToolPluginResult on success."""
        runner = SuccessRunner(cache_dir=tmp_path)
        service = ToolService(runner)

        result = asyncio.run(service.run_plugin("pyright", repo_root=tmp_path))

        expect_is_instance(result, ToolPluginResult)
