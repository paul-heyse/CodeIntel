"""High-level façade around ToolRunner for external CLI integrations.

This module provides a simplified interface to tool plugins. The plugins
handle all parsing internally; this service delegates to them and returns
the parsed domain objects.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.tools import (
    ToolPlugin,
    ToolPluginRegistry,
    ToolPluginResult,
    ToolStatus,
    build_default_registry,
)
from codeintel.ingestion.tools.results import (
    CoverageFileSummary,
    CoverageReport,
    DiagnosticReport,
    ScipIndexResult,
    TestReport,
)

log = logging.getLogger(__name__)


def _unlink_missing(path: Path) -> None:
    path.unlink(missing_ok=True)


def _path_is_file(path: Path) -> bool:
    return path.is_file()


@dataclass(frozen=True)
class CoverageFileReport:
    """Normalized coverage summary for a single file.

    This dataclass provides backward compatibility with existing code
    that expects this interface from ToolService.
    """

    rel_path: str
    executed_lines: set[int]
    missing_lines: set[int]

    @classmethod
    def from_summary(cls, summary: CoverageFileSummary) -> CoverageFileReport:
        """
        Convert from the new CoverageFileSummary domain type.

        Parameters
        ----------
        summary
            CoverageFileSummary from tool plugin.

        Returns
        -------
        CoverageFileReport
            Converted report with mutable sets.
        """
        return cls(
            rel_path=summary.rel_path,
            executed_lines=set(summary.executed_lines),
            missing_lines=set(summary.missing_lines),
        )


class ToolService:
    """Orchestrate external tooling via tool plugins.

    This service is a thin façade that delegates to tool plugins. The plugins
    handle execution and parsing; this service provides a simple interface
    for callers.
    """

    def __init__(self, runner: ToolRunner, tools_config: ToolsConfig | None = None) -> None:
        self.runner = runner
        self.tools_config = tools_config or runner.tools_config
        self._plugins: ToolPluginRegistry = build_default_registry(self.runner, self.tools_config)

    def get_plugin(self, name: str) -> ToolPlugin:
        """
        Return a registered plugin by name.

        Parameters
        ----------
        name
            Plugin registry name.

        Returns
        -------
        ToolPlugin
            Registered plugin instance.
        """
        return self._plugins.get(name)

    async def run_plugin(self, name: str, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """
        Execute a tool plugin by name and return its normalized result.

        Parameters
        ----------
        name
            Plugin registry name (for example, "pyright", "coverage", "scip").
        repo_root
            Repository root passed to the plugin.
        **kwargs
            Plugin-specific arguments (for example, repo_root, output_path).

        Returns
        -------
        ToolPluginResult
            Normalized plugin result including status and parsed output.

        Raises
        ------
        KeyError
            If no plugin is registered under the provided name.
        """
        if name not in self._plugins.names():
            raise KeyError(name)
        plugin = self.get_plugin(name)
        return await plugin.run(repo_root=repo_root, **kwargs)

    async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyright and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyright invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.

        Raises
        ------
        ToolExecutionError
            Raised when pyright exits with an unexpected status.
        RuntimeError
            Raised when a plugin result is missing the expected run metadata.
        """
        plugin_result = await self.run_plugin("pyright", repo_root=repo_root)

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return {}

        if plugin_result.status is not ToolStatus.OK:
            err = plugin_result.error
            if isinstance(err, ToolExecutionError):
                raise err
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            message = "pyright plugin failed without ToolRunResult"
            raise RuntimeError(message)

        # Return parsed diagnostics from plugin
        parsed = plugin_result.parsed
        if isinstance(parsed, DiagnosticReport):
            return parsed.errors_by_path()

        # Fallback for backward compatibility
        if plugin_result.run is None:
            message = "pyright plugin returned no run metadata"
            raise RuntimeError(message)
        return {}

    async def run_pyrefly(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyrefly and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyrefly invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.
        """
        output_path = self.runner.cache_dir / "pyrefly.json"

        plugin_result = await self.run_plugin(
            "pyrefly",
            repo_root=repo_root,
            output_path=output_path,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("pyrefly binary not found; treating all files as 0 errors")
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        if plugin_result.status is not ToolStatus.OK:
            log.warning(
                "pyrefly invocation failed or produced unusable output; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        # Clean up temp file
        json_path = plugin_result.artifacts.get("pyrefly_json", output_path)
        await to_thread.run_sync(_unlink_missing, json_path)

        # Return parsed diagnostics from plugin
        parsed = plugin_result.parsed
        if isinstance(parsed, DiagnosticReport):
            return parsed.errors_by_path()

        return {}

    async def run_ruff(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run ruff and return lint error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the ruff invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to lint error counts.

        Raises
        ------
        ToolExecutionError
            Raised when ruff exits with an unexpected status.
        RuntimeError
            Raised when a plugin result is missing the expected run metadata.
        """
        plugin_result = await self.run_plugin(
            "ruff",
            repo_root=repo_root,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return {}

        if plugin_result.status is not ToolStatus.OK:
            err = plugin_result.error
            if isinstance(err, ToolExecutionError):
                raise err
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            message = "ruff plugin failed without ToolRunResult"
            raise RuntimeError(message)

        # Return parsed diagnostics from plugin
        parsed = plugin_result.parsed
        if isinstance(parsed, DiagnosticReport):
            return parsed.errors_by_path()

        if plugin_result.run is None:
            message = "ruff plugin returned no run metadata"
            raise RuntimeError(message)
        return {}

    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CoverageFileReport]:
        """
        Run coverage json export and return normalized file reports.

        Parameters
        ----------
        repo_root
            Repository root supplied to the coverage invocation.
        coverage_file
            Optional explicit .coverage path to read from.
        output_path
            Optional path where the JSON output should be written.

        Returns
        -------
        list[CoverageFileReport]
            Normalized coverage summaries grouped per file.
        """
        target_output = output_path or (self.runner.cache_dir / "coverage.json")
        data_file = coverage_file or self.tools_config.coverage_file

        plugin_result = await self.run_plugin(
            "coverage",
            repo_root=repo_root,
            coverage_file=data_file,
            output_path=target_output,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("coverage binary not found; skipping coverage ingestion")
            await to_thread.run_sync(_unlink_missing, target_output)
            return []

        if plugin_result.status is not ToolStatus.OK:
            log.warning(
                "coverage CLI failed or returned non-zero exit; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            await to_thread.run_sync(_unlink_missing, target_output)
            return []

        # Clean up temp file
        json_path = plugin_result.artifacts.get("coverage_json", target_output)
        await to_thread.run_sync(_unlink_missing, json_path)

        # Convert parsed CoverageReport to legacy CoverageFileReport list
        parsed = plugin_result.parsed
        if isinstance(parsed, CoverageReport):
            return [CoverageFileReport.from_summary(f) for f in parsed.files]

        return []

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """
        Generate a pytest JSON report when missing.

        Parameters
        ----------
        repo_root
            Repository root passed to the pytest invocation.
        json_report_path
            Output path for the pytest JSON report.

        Returns
        -------
        bool
            True when pytest was executed to produce the report, False when reused.

        Raises
        ------
        ToolExecutionError
            Raised when pytest execution fails or does not create a report.
        ToolNotFoundError
            Raised when the pytest binary cannot be resolved.
        RuntimeError
            Raised when a plugin result is missing the expected run metadata.
        """
        if await to_thread.run_sync(_path_is_file, json_report_path):
            return False

        plugin_result = await self.run_plugin(
            "pytest",
            repo_root=repo_root,
            json_report_path=json_report_path,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            raise ToolNotFoundError(ToolName.PYTEST, self.tools_config.pytest_bin)

        if plugin_result.status is not ToolStatus.OK:
            err = plugin_result.error
            if isinstance(err, ToolExecutionError):
                raise err
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            message = "pytest plugin failed without ToolRunResult"
            raise RuntimeError(message)

        if plugin_result.run is None:
            message = "pytest plugin returned no run metadata"
            raise RuntimeError(message)
        exists = await to_thread.run_sync(_path_is_file, json_report_path)
        if not exists:
            raise ToolExecutionError(plugin_result.run)
        return True

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """
        Run scip-python for a full index and export to JSON.

        Returns
        -------
        ScipIndexResult
            Parsed SCIP index result.

        Raises
        ------
        ToolExecutionError
            Raised when SCIP tooling exits with an error.
        ToolNotFoundError
            Raised when SCIP tooling binaries cannot be resolved.
        RuntimeError
            Raised when plugin results are missing required metadata.
        """
        plugin_result = await self.run_plugin(
            "scip",
            repo_root=repo_root,
            output_scip=output_scip,
            output_json=output_json,
            target_dir=target_dir,
            rel_paths=None,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            error = plugin_result.error
            if isinstance(error, ToolNotFoundError):
                raise error
            configured_path = self.tools_config.resolve_path(plugin_result.tool)
            raise ToolNotFoundError(plugin_result.tool, configured_path)

        if plugin_result.status is not ToolStatus.OK:
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            message = "SCIP plugin failed without ToolRunResult"
            raise RuntimeError(message)

        # Return parsed SCIP result
        parsed = plugin_result.parsed
        if isinstance(parsed, ScipIndexResult):
            return parsed

        return ScipIndexResult.empty()

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: list[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """
        Run scip-python for a subset of files and export to JSON.

        Returns
        -------
        ScipIndexResult
            Parsed SCIP index result for the shard.

        Raises
        ------
        ToolExecutionError
            Raised when SCIP tooling exits with an error.
        ToolNotFoundError
            Raised when SCIP tooling binaries cannot be resolved.
        RuntimeError
            Raised when plugin results are missing required metadata.
        """
        plugin_result = await self.run_plugin(
            "scip",
            repo_root=repo_root,
            output_scip=output_scip,
            output_json=output_json,
            target_dir=target_dir,
            rel_paths=rel_paths,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            error = plugin_result.error
            if isinstance(error, ToolNotFoundError):
                raise error
            configured_path = self.tools_config.resolve_path(plugin_result.tool)
            raise ToolNotFoundError(plugin_result.tool, configured_path)

        if plugin_result.status is not ToolStatus.OK:
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            message = "SCIP plugin failed without ToolRunResult"
            raise RuntimeError(message)

        # Return parsed SCIP result
        parsed = plugin_result.parsed
        if isinstance(parsed, ScipIndexResult):
            return parsed

        return ScipIndexResult.empty()

    @staticmethod
    def get_test_report(result: ToolPluginResult) -> TestReport:
        """
        Extract parsed TestReport from a pytest plugin result.

        Parameters
        ----------
        result
            ToolPluginResult from pytest plugin execution.

        Returns
        -------
        TestReport
            Parsed test report or empty report.
        """
        parsed = result.parsed
        if isinstance(parsed, TestReport):
            return parsed
        return TestReport.empty()
