"""High-level façade around ToolRunner for external CLI integrations.

This module provides a simplified interface to tool plugins. The plugins
handle all parsing internally; this service delegates to them and returns
the parsed domain objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anyio import to_thread

from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
)
from codeintel.ingestion.engine.plugins import (
    ToolStatus,
    build_default_registry,
)
from codeintel.ingestion.engine.results import (
    CoverageReport,
    DiagnosticReport,
    ScipIndexResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import (
        ToolRunner,
    )
    from codeintel.ingestion.engine.plugins import (
        ToolPlugin,
        ToolPluginRegistry,
        ToolPluginResult,
    )

log = logging.getLogger(__name__)


def _unlink_missing(path: Path) -> None:
    path.unlink(missing_ok=True)


def _path_is_file(path: Path) -> bool:
    return path.is_file()


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

        parsed = plugin_result.parsed
        if isinstance(parsed, DiagnosticReport):
            return parsed.errors_by_path()

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

        json_path = plugin_result.artifacts.get("pyrefly_json", output_path)
        await to_thread.run_sync(_unlink_missing, json_path)

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

        parsed = plugin_result.parsed
        if isinstance(parsed, DiagnosticReport):
            return parsed.errors_by_path()

        if plugin_result.run is None:
            message = "ruff plugin returned no run metadata"
            raise RuntimeError(message)
        return {}

    async def run_coverage_report(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageReport:
        """Run coverage JSON export and return a CoverageReport.

        Parameters
        ----------
        repo_root
            Repository root directory.
        coverage_file
            Optional explicit coverage data file path.
        output_path
            Optional path for JSON output; defaults to a cache location.

        Returns
        -------
        CoverageReport
            Parsed coverage data for all files. Returns CoverageReport.empty()
            when the coverage tool is missing or fails.
        """
        target_output = output_path or (self.runner.cache_dir / "coverage.json")
        data_file = coverage_file or self.tools_config.coverage_file

        plugin_result = await self.run_plugin(
            "coverage",
            repo_root=repo_root,
            coverage_file=data_file,
            output_path=target_output,
        )

        json_path = plugin_result.artifacts.get("coverage_json", target_output)
        await to_thread.run_sync(_unlink_missing, json_path)

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return CoverageReport.empty()

        if plugin_result.status is not ToolStatus.OK:
            log.warning(
                "coverage CLI failed or returned non-zero exit; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            return CoverageReport.empty()

        parsed = plugin_result.parsed
        if isinstance(parsed, CoverageReport):
            return parsed

        log.warning(
            "coverage plugin returned unexpected parsed payload type: %r",
            type(parsed),
        )
        return CoverageReport.empty()

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
        output_json = output_scip.with_suffix(".json")
        plugin_result = await self.run_plugin(
            "scip",
            repo_root=repo_root,
            output_scip=output_scip,
            output_json=output_json,
            target_dir=None,
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

        parsed = plugin_result.parsed
        if isinstance(parsed, ScipIndexResult):
            return parsed

        return ScipIndexResult.empty()
