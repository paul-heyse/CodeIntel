"""Pyrefly plugin for the ingestion tool runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from anyio import to_thread

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


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class PyreflyPlugin(ToolPlugin):
    """Plugin responsible for running pyrefly and normalizing failures."""

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
        Invoke pyrefly with JSON output and normalize outcomes.

        The plugin mirrors the existing ToolService semantics by degrading to
        empty results instead of raising on non-OK exits when no report exists.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the pyrefly plugin.

        Raises
        ------
        TypeError
            Raised when required keyword arguments are missing or of the wrong type.
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
                status=ToolStatus.ERROR,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
            )

        artifacts = {"pyrefly_json": output_path}
        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts=artifacts,
            run=result,
            error=None,
        )
