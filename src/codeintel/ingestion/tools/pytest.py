"""Pytest plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.infrastructure_utilities.types import ToolStatus
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
)
from codeintel.ingestion.tools.results import TestReport

log = logging.getLogger(__name__)


def _parse_pytest_json(
    payload: Mapping[str, Any],
    report_path: Path | None = None,
) -> TestReport:
    """
    Parse pytest-json-report output into a TestReport.

    Parameters
    ----------
    payload
        Parsed JSON from pytest-json-report.
    report_path
        Path to the JSON file for reference.

    Returns
    -------
    TestReport
        Parsed test results.
    """
    # pytest-json-report has tests at top-level "tests" key
    tests = payload.get("tests")
    if tests is None and "report" in payload:
        tests = payload["report"].get("tests")

    if not isinstance(tests, list):
        return TestReport.empty()

    return TestReport.from_test_entries(tests, report_path=report_path)


@dataclass
class PytestPlugin(ToolPlugin):
    """Plugin for running pytest and parsing the JSON report."""

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
        Execute pytest and return parsed test report.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed TestReport.

        Raises
        ------
        TypeError
            Raised when required keyword arguments are missing or of wrong type.
        """
        json_report_path_obj = kwargs.get("json_report_path")
        if not isinstance(json_report_path_obj, Path):
            message = "pytest plugin requires json_report_path of type Path"
            raise TypeError(message)
        json_report_path = json_report_path_obj

        await to_thread.run_sync(lambda: json_report_path.parent.mkdir(parents=True, exist_ok=True))

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
                parsed=TestReport.empty(),
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.FAILED,
                artifacts={"pytest_json_report": json_report_path},
                run=exc.result,
                error=exc,
                parsed=TestReport.empty(),
            )

        # Parse the JSON output file
        parsed = TestReport.empty()

        def _is_file() -> bool:
            return json_report_path.is_file()

        output_exists = await to_thread.run_sync(_is_file)
        if output_exists:

            def _load_and_parse() -> TestReport:
                try:
                    payload = json.loads(json_report_path.read_text(encoding="utf-8"))
                    return _parse_pytest_json(payload, json_report_path)
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("Failed to parse pytest JSON: %s", exc)
                    return TestReport.empty()

            parsed = await to_thread.run_sync(_load_and_parse)

        status = ToolStatus.OK if result.ok else ToolStatus.FAILED
        artifacts = {"pytest_json_report": json_report_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
            parsed=parsed,
        )
