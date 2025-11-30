"""Pyright plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class PyrightPlugin(ToolPlugin):
    """Plugin responsible for running pyright and normalizing failures."""

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
        Invoke pyright with --outputjson and normalize outcomes.

        Returns a ToolPluginResult that never raises; ToolService can decide
        whether to downgrade or re-raise based on status.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the pyright plugin.
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
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.ERROR,
                artifacts={},
                run=exc.result,
                error=exc,
            )

        if result.returncode not in {0, 1}:
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.ERROR,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
            )

        status = ToolStatus.OK
        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts={},
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )

    @staticmethod
    def parse_diagnostics(result: ToolRunResult) -> dict[str, int]:
        """
        Parse pyright JSON from stdout into path -> error_count mapping.

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
