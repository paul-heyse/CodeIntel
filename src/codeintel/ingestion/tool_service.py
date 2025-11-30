"""High-level façade around ToolRunner for external CLI integrations."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.paths import normalize_rel_path, repo_relpath
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

log = logging.getLogger(__name__)


def _unlink_missing(path: Path) -> None:
    path.unlink(missing_ok=True)


def _path_is_file(path: Path) -> bool:
    return path.is_file()


@dataclass(frozen=True)
class CoverageFileReport:
    """Normalized coverage summary for a single file."""

    rel_path: str
    executed_lines: set[int]
    missing_lines: set[int]


class ToolService:
    """Orchestrate external tooling and parse outputs for ingestion modules."""

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
            Normalized plugin result including status and artifacts.

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

        if plugin_result.run is None:
            message = "pyright plugin returned no run metadata"
            raise RuntimeError(message)
        return _parse_pyright_errors(plugin_result.run.stdout, repo_root)

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
        payload = await to_thread.run_sync(ToolRunner.load_json, json_path) or {}
        await to_thread.run_sync(_unlink_missing, json_path)
        return _parse_pyrefly_errors(payload, repo_root)

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

        if plugin_result.run is None:
            message = "ruff plugin returned no run metadata"
            raise RuntimeError(message)
        return _parse_ruff_errors(plugin_result.run.stdout, repo_root)

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

        json_path = plugin_result.artifacts.get("coverage_json", target_output)
        payload = await to_thread.run_sync(ToolRunner.load_json, json_path) or {}
        await to_thread.run_sync(_unlink_missing, json_path)
        return _parse_coverage_payload(payload, repo_root)

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
    ) -> None:
        """
        Run scip-python for a full index and export to JSON.

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

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: Sequence[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """
        Run scip-python for a subset of files and export to JSON.

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
            rel_paths=list(rel_paths),
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


def _parse_pyrefly_errors(payload: Mapping[str, Any], repo_root: Path) -> dict[str, int]:
    errors_field = payload.get("errors") if isinstance(payload, Mapping) else None
    errors: Iterable[Mapping[str, Any]] = errors_field if isinstance(errors_field, list) else []
    counts: dict[str, int] = {}
    for diag in errors:
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("path")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_pyright_errors(stdout: str, repo_root: Path) -> dict[str, int]:
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse pyright JSON output: %s", exc)
        return {}

    diagnostics = payload.get("generalDiagnostics") if isinstance(payload, dict) else None
    if not isinstance(diagnostics, list):
        return {}

    counts: dict[str, int] = {}
    for diag in diagnostics:
        if not isinstance(diag, Mapping):
            continue
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("file")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_ruff_errors(stdout: str, repo_root: Path) -> dict[str, int]:
    try:
        payload = json.loads(stdout) if stdout else []
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse ruff JSON output: %s", exc)
        return {}
    if not isinstance(payload, list):
        return {}

    counts: dict[str, int] = {}
    for diag in payload:
        if not isinstance(diag, Mapping):
            continue
        file_name = diag.get("filename")
        if not file_name:
            continue
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1
    return counts


def _parse_coverage_payload(
    payload: Mapping[str, Any],
    repo_root: Path,
) -> list[CoverageFileReport]:
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping):
        return []

    reports: list[CoverageFileReport] = []
    for file_name, data in files.items():
        if not isinstance(data, Mapping):
            continue
        executed = {int(line) for line in data.get("executed_lines", []) if isinstance(line, int)}
        missing = {int(line) for line in data.get("missing_lines", []) if isinstance(line, int)}
        rel_path = _safe_relpath(repo_root, Path(str(file_name)))
        if rel_path is None:
            continue
        reports.append(
            CoverageFileReport(
                rel_path=rel_path,
                executed_lines=executed,
                missing_lines=missing,
            )
        )
    return reports


def _safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    try:
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None
