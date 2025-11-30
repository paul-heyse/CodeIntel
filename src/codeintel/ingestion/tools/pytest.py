"""Pytest plugin for the ingestion tool runtime."""

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
class PytestPlugin(ToolPlugin):
    """Plugin for running pytest with the JSON-report plugin enabled."""

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = field(
        default_factory=lambda: ToolPluginMetadata(
            name="pytest",
            produces_artifacts=("pytest_json_report",),
            consumes_configs=("pytest_bin",),
            datasets=("analytics.test_catalog",),
        )
    )

    async def run(self, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """
        Execute pytest and write a JSON report to the requested path.

        Returns
        -------
        ToolPluginResult
            Normalized execution result from the pytest plugin.

        Raises
        ------
        TypeError
            Raised when required keyword arguments are missing or of the wrong type.
        """
        json_report_path_obj = kwargs.get("json_report_path")
        if not isinstance(json_report_path_obj, Path):
            message = "pytest plugin requires json_report_path of type Path"
            raise TypeError(message)
        json_report_path = json_report_path_obj

        await to_thread.run_sync(
            lambda: json_report_path.parent.mkdir(parents=True, exist_ok=True)
        )

        args = [
            "-q",
            "--disable-warnings",
            "--maxfail=1",
            "--json-report",
            f"--json-report-file={json_report_path}",
        ]

        try:
            result = await self.runner.run_async(
                ToolName.PYTEST,
                args,
                cwd=repo_root,
                output_path=json_report_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("pytest binary not found; skipping test ingestion")
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.ERROR,
                artifacts={"pytest_json_report": json_report_path},
                run=exc.result,
                error=exc,
            )

        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        artifacts = {"pytest_json_report": json_report_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )
