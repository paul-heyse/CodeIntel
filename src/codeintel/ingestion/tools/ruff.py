"""Ruff plugin for the ingestion tool runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
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

log = logging.getLogger(__name__)


@dataclass
class RuffPlugin(ToolPlugin):
    """
    Plugin responsible for running ruff and normalizing failures.

    This plugin does not parse diagnostics itself; ToolService.run_ruff still
    calls _parse_ruff_errors so callers keep the same behaviour.
    """

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
        Invoke ruff with JSON output and normalize the outcome.

        Returns a ToolPluginResult that does not raise; ToolService decides
        whether to downgrade or raise ToolExecutionError.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the ruff plugin.
        """
        try:
            result = await self.runner.run_async(
                ToolName.RUFF,
                ["check", str(repo_root), "--output-format", "json"],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return ToolPluginResult(
                tool=ToolName.RUFF,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )

        if result.returncode not in {0, 1}:
            err = ToolExecutionError(result)
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.ERROR,
                artifacts={},
                run=result,
                error=err,
            )

        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts={},
            run=result,
            error=None,
        )
