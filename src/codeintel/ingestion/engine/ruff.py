"""Ruff plugin for the ingestion tool runtime."""

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
    from codeintel.ingestion.engine.infrastructure import (
        ToolRunner,
    )

log = logging.getLogger(__name__)


def _parse_ruff_output(
    stdout: str,
    repo_root: Path,
) -> DiagnosticReport:
    """
    Parse ruff JSON output into a DiagnosticReport.

    Parameters
    ----------
    stdout
        Raw JSON array output from ruff check --output-format json.
    repo_root
        Repository root for path normalization.

    Returns
    -------
    DiagnosticReport
        Parsed diagnostic counts per file.
    """
    if not stdout.strip():
        return DiagnosticReport.empty("ruff")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse ruff JSON output: %s", exc)
        return DiagnosticReport.empty("ruff")

    if not isinstance(payload, list):
        return DiagnosticReport.empty("ruff")

    counts: dict[str, int] = {}
    for diag in payload:
        if not isinstance(diag, Mapping):
            continue
        file_name = diag.get("filename")
        if not file_name:
            continue
        try:
            rel_path = repo_relpath(repo_root, Path(str(file_name)))
        except ValueError:
            continue
        counts[rel_path] = counts.get(rel_path, 0) + 1

    return DiagnosticReport.from_error_counts("ruff", counts, raw_output=stdout)


@dataclass
class RuffPlugin(DiagnosticToolPlugin):
    """Plugin responsible for running ruff and parsing diagnostics."""

    tool_name: ClassVar[ToolName] = ToolName.RUFF
    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="ruff",
            produces_artifacts=(),
            consumes_configs=("ruff_bin",),
            datasets=("analytics.static_diagnostics",),
        )
    )

    async def run(self, *, repo_root: Path, **_: object) -> ToolPluginResult:
        """
        Invoke ruff with JSON output and return parsed diagnostics.

        Returns a ToolPluginResult with parsed DiagnosticReport.
        On failure, returns empty diagnostics rather than raising.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed diagnostics.
        """
        try:
            result = await self.runner.run_async(
                ToolName.RUFF,
                ["check", str(repo_root), "--output-format", "json"],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return self._not_found_result()

        if result.returncode not in {0, 1}:
            err = ToolExecutionError(result)
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.FAILED,
                artifacts={},
                run=result,
                error=err,
                parsed=DiagnosticReport.empty("ruff"),
            )

        parsed = _parse_ruff_output(result.stdout, repo_root)

        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts={},
            run=result,
            error=None,
            parsed=parsed,
        )
