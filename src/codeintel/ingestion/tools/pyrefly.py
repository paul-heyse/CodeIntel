"""Pyrefly plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.infrastructure.paths import normalize_rel_path, repo_relpath
from codeintel.ingestion.tools.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.tools.results import DiagnosticReport

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _parse_pyrefly_output(
    payload: Mapping[str, Any],
    repo_root: Path,
) -> DiagnosticReport:
    """
    Parse pyrefly JSON output into a DiagnosticReport.

    Parameters
    ----------
    payload
        Parsed JSON from pyrefly output file.
    repo_root
        Repository root for path normalization.

    Returns
    -------
    DiagnosticReport
        Parsed diagnostic counts per file.
    """
    errors_field = payload.get("errors") if isinstance(payload, Mapping) else None
    errors: list[Mapping[str, Any]] = errors_field if isinstance(errors_field, list) else []

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

    return DiagnosticReport.from_error_counts("pyrefly", counts)


@dataclass
class PyreflyPlugin(ToolPlugin):
    """Plugin responsible for running pyrefly and parsing diagnostics."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="pyrefly",
            produces_artifacts=("pyrefly_json",),
            consumes_configs=("pyrefly_bin",),
            datasets=("analytics.static_diagnostics", "analytics.typedness"),
        )
    )

    async def run(self, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """
        Invoke pyrefly with JSON output and return parsed diagnostics.

        Returns a ToolPluginResult with parsed DiagnosticReport.
        The plugin degrades to empty results on failures.

        Parameters
        ----------
        repo_root
            Repository root to analyze.
        **kwargs
            Must include output_path: Path for JSON output.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed diagnostics.

        Raises
        ------
        TypeError
            Raised when required keyword arguments are missing or of wrong type.
        """
        output_path_obj = kwargs.get("output_path")
        if not isinstance(output_path_obj, Path):
            message = "pyrefly plugin requires an output_path of type Path"
            raise TypeError(message)
        output_path = output_path_obj

        await to_thread.run_sync(_mkdir_parents, output_path.parent)

        args = [
            "check",
            str(repo_root),
            "--output-format",
            "json",
            "--output",
            str(output_path),
            "--summary",
            "none",
            "--count-errors=0",
        ]

        try:
            result = await self.runner.run_async(
                ToolName.PYREFLY,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            return ToolPluginResult(
                tool=ToolName.PYREFLY,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
                parsed=DiagnosticReport.empty("pyrefly"),
            )

        def _is_file() -> bool:
            return output_path.is_file()

        output_exists = await to_thread.run_sync(_is_file)
        if not output_exists and result.returncode != 0:
            log.warning(
                "pyrefly exited with code %s and produced no output; stdout=%s stderr=%s",
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.FAILED,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
                parsed=DiagnosticReport.empty("pyrefly"),
            )

        # Parse the JSON output file
        parsed = DiagnosticReport.empty("pyrefly")
        if output_exists:

            def _load_json() -> dict[str, object]:
                try:
                    return json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("Failed to parse pyrefly JSON: %s", exc)
                    return {}

            payload = await to_thread.run_sync(_load_json)
            parsed = _parse_pyrefly_output(payload, repo_root)

        artifacts = {"pyrefly_json": output_path}
        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts=artifacts,
            run=result,
            error=None,
            parsed=parsed,
        )
