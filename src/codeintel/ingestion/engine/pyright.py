"""Pyright plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.engine.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.engine.results import DiagnosticReport
from codeintel.ingestion.infrastructure.paths import normalize_rel_path, repo_relpath

log = logging.getLogger(__name__)


def _safe_relpath(repo_root: Path, file_path: Path) -> str | None:
    """
    Safely compute repository-relative path.

    Parameters
    ----------
    repo_root
        Repository root path.
    file_path
        Absolute or relative file path.

    Returns
    -------
    str | None
        Normalized relative path or None on failure.
    """
    try:
        candidate = file_path if file_path.is_absolute() else repo_root / file_path
        return normalize_rel_path(repo_relpath(repo_root, candidate))
    except ValueError:
        return None


def _parse_pyright_output(
    stdout: str,
    repo_root: Path,
) -> DiagnosticReport:
    """
    Parse pyright JSON output into a DiagnosticReport.

    Parameters
    ----------
    stdout
        Raw JSON output from pyright --outputjson.
    repo_root
        Repository root for path normalization.

    Returns
    -------
    DiagnosticReport
        Parsed diagnostic counts per file.
    """
    if not stdout.strip():
        return DiagnosticReport.empty("pyright")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse pyright JSON output: %s", exc)
        return DiagnosticReport.empty("pyright")

    if not isinstance(payload, dict):
        return DiagnosticReport.empty("pyright")

    diagnostics = payload.get("generalDiagnostics")
    if not isinstance(diagnostics, list):
        return DiagnosticReport.empty("pyright")

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

    return DiagnosticReport.from_error_counts(
        "pyright",
        counts,
        raw_output=stdout,
    )


@dataclass
class PyrightPlugin(ToolPlugin):
    """Plugin responsible for running pyright and parsing diagnostics."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="pyright",
            produces_artifacts=(),
            consumes_configs=("pyright_bin",),
            datasets=("analytics.typedness", "analytics.static_diagnostics"),
        )
    )

    async def run(self, *, repo_root: Path, **_: object) -> ToolPluginResult:
        """
        Invoke pyright with --outputjson and return parsed diagnostics.

        Returns a ToolPluginResult with parsed DiagnosticReport.
        On failure, returns empty diagnostics rather than raising.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed diagnostics.
        """
        try:
            result = await self.runner.run_async(
                ToolName.PYRIGHT,
                ["--outputjson", str(repo_root)],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
                parsed=DiagnosticReport.empty("pyright"),
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.FAILED,
                artifacts={},
                run=exc.result,
                error=exc,
                parsed=DiagnosticReport.empty("pyright"),
            )

        # pyright returns 0 on success, 1 when there are errors
        if result.returncode not in {0, 1}:
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.FAILED,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
                parsed=DiagnosticReport.empty("pyright"),
            )

        # Parse diagnostics from stdout
        parsed = _parse_pyright_output(result.stdout, repo_root)

        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts={},
            run=result,
            error=None,
            parsed=parsed,
        )

    @staticmethod
    def parse_diagnostics(result: ToolRunResult) -> dict[str, int]:
        """
        Parse pyright JSON from stdout into path -> error_count mapping.

        Deprecated: Use the parsed field on ToolPluginResult instead.

        Parameters
        ----------
        result
            ToolRunResult containing pyright stdout JSON.

        Returns
        -------
        dict[str, int]
            Mapping from file path to error count.

        Raises
        ------
        ToolExecutionError
            Raised when stdout is not valid JSON.
        """
        if not result.stdout.strip():
            return {}

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(result) from exc

        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            log.warning("Unexpected pyright JSON structure; missing 'summary'")
            return {}

        files_field = summary.get("files", {})
        if not isinstance(files_field, dict):
            return {}

        errors: dict[str, int] = {}
        for path, info in files_field.items():
            if not isinstance(info, dict):
                continue
            count = int(info.get("errorCount", 0))
            errors[str(path)] = count
        return errors
