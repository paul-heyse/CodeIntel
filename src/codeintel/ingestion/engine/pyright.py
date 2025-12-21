"""Pyright plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.core.paths import repo_relpath
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.engine.plugins import (
    DiagnosticToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.engine.results import DiagnosticReport

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import ToolRunner

log = logging.getLogger(__name__)


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
        try:
            rel_path = repo_relpath(repo_root, Path(str(file_name)))
        except ValueError:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1

    return DiagnosticReport.from_error_counts(
        "pyright",
        counts,
        raw_output=stdout,
    )


@dataclass
class PyrightPlugin(DiagnosticToolPlugin):
    """Plugin responsible for running pyright and parsing diagnostics."""

    tool_name: ClassVar[ToolName] = ToolName.PYRIGHT
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
                options=ToolRunOptions(
                    cwd=repo_root,
                    timeout_s=self.tools_config.default_timeout_s,
                ),
            )
        except ToolNotFoundError:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return self._not_found_result()
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.FAILED,
                artifacts={},
                run=exc.result,
                error=exc,
                parsed=DiagnosticReport.empty("pyright"),
            )

        if result.returncode not in {0, 1}:
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.FAILED,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
                parsed=DiagnosticReport.empty("pyright"),
            )

        parsed = _parse_pyright_output(result.stdout, repo_root)

        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts={},
            run=result,
            error=None,
            parsed=parsed,
        )
