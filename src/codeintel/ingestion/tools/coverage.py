"""Coverage plugin for the ingestion tool runtime."""

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


@dataclass
class CoveragePlugin(ToolPlugin):
    """
    Plugin for running `coverage json` to produce a JSON coverage report.

    The plugin ensures the output directory exists and normalizes errors; it
    does not parse the JSON, leaving that to ToolService.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="coverage",
            produces_artifacts=("coverage_json",),
            consumes_configs=("coverage_bin",),
            datasets=("analytics.coverage_lines",),
        )
    )

    async def run(
        self,
        *,
        repo_root: Path,
        coverage_file: Path | None,
        output_path: Path,
        **_: object,
    ) -> ToolPluginResult:
        """
        Run coverage CLI to produce a JSON report.

        Parameters
        ----------
        repo_root:
            Repository root passed to the CLI via `cwd`.
        coverage_file:
            Path to `.coverage` data file.
        output_path:
            Target JSON file path.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the coverage plugin.
        """
        await to_thread.run_sync(
            lambda: output_path.parent.mkdir(parents=True, exist_ok=True)
        )

        args = ["json", "--quiet", "-o", str(output_path)]
        if coverage_file is not None:
            args.append(f"--data-file={coverage_file}")
        try:
            result = await self.runner.run_async(
                ToolName.COVERAGE,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.ERROR,
                artifacts={"coverage_json": output_path},
                run=exc.result,
                error=exc,
            )

        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        artifacts = {"coverage_json": output_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )
