"""Pytest plugin for the ingestion tool runtime."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anyio import to_thread

from codeintel.ingestion.engine.capabilities import ToolCapability, ToolCapabilityProbe
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunOptions,
)
from codeintel.ingestion.engine.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)
from codeintel.ingestion.engine.results import TestReport

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.engine.infrastructure import (
        ToolRunner,
        ToolRunResult,
    )

log = logging.getLogger(__name__)

_PYTEST_NO_TESTS_EXIT_CODE = 5


@dataclass(frozen=True)
class _PytestRunOutcome:
    status: ToolStatus
    run: ToolRunResult | None
    error: Exception | None
    reason: str | None = None


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
    _capability_probe: ToolCapabilityProbe | None = field(
        default=None,
        init=False,
        repr=False,
    )

    async def run(self, *, repo_root: Path, **kwargs: object) -> ToolPluginResult:
        """
        Execute pytest and return parsed test report.

        Returns
        -------
        ToolPluginResult
            Normalized execution result with parsed TestReport.
        """
        json_report_path = _require_json_report_path(kwargs)
        capability = await self._resolve_json_report_capability(repo_root)
        capability_result = self._capability_result(capability)
        if capability_result is not None:
            return capability_result

        return await self._run_pytest_with_report(repo_root, json_report_path)

    async def _resolve_json_report_capability(self, repo_root: Path) -> ToolCapability:
        probe = self._capability_probe
        if probe is None:
            probe = ToolCapabilityProbe(runner=self.runner, tools_config=self.tools_config)
            self._capability_probe = probe
        return await probe.pytest_json_report(repo_root=repo_root)

    def _capability_result(self, capability: ToolCapability) -> ToolPluginResult | None:
        if capability.state == "missing_tool":
            exc = ToolNotFoundError(ToolName.PYTEST, self.tools_config.pytest_bin)
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
                parsed=TestReport.empty(),
            )
        if capability.state == "unsupported":
            log.warning("pytest json-report support missing; skipping test ingestion")
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.SKIPPED,
                artifacts={},
                run=None,
                error=None,
                parsed=TestReport.empty(),
            )
        return None

    async def _run_pytest_with_report(
        self,
        repo_root: Path,
        json_report_path: Path,
    ) -> ToolPluginResult:
        await to_thread.run_sync(lambda: json_report_path.parent.mkdir(parents=True, exist_ok=True))
        args = [
            "-q",
            "--disable-warnings",
            "--maxfail=1",
            "--json-report",
            f"--json-report-file={json_report_path}",
        ]
        outcome = await self._run_pytest(repo_root, json_report_path, args)
        if outcome.status is ToolStatus.NOT_FOUND:
            error = outcome.error or ToolNotFoundError(
                ToolName.PYTEST,
                self.tools_config.pytest_bin,
            )
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=error,
                parsed=TestReport.empty(),
            )
        if outcome.status is ToolStatus.SKIPPED:
            if outcome.reason == "no_tests":
                log.info("pytest run collected no tests; skipping report ingestion")
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.SKIPPED,
                artifacts={"pytest_json_report": json_report_path},
                run=outcome.run,
                error=None,
                parsed=TestReport.empty(),
            )
        if outcome.status is ToolStatus.FAILED:
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.FAILED,
                artifacts={"pytest_json_report": json_report_path},
                run=outcome.run,
                error=outcome.error,
                parsed=TestReport.empty(),
            )
        parsed = await _load_pytest_report(json_report_path)
        return ToolPluginResult(
            tool=ToolName.PYTEST,
            status=ToolStatus.OK,
            artifacts={"pytest_json_report": json_report_path},
            run=outcome.run,
            error=None,
            parsed=parsed,
        )

    async def _run_pytest(
        self,
        repo_root: Path,
        json_report_path: Path,
        args: list[str],
    ) -> _PytestRunOutcome:
        try:
            result = await self.runner.run_async(
                ToolName.PYTEST,
                args,
                options=ToolRunOptions(
                    cwd=repo_root,
                    output_path=json_report_path,
                    timeout_s=self.tools_config.default_timeout_s,
                ),
            )
        except ToolNotFoundError:
            log.warning("pytest binary not found; skipping test ingestion")
            outcome = _PytestRunOutcome(
                status=ToolStatus.NOT_FOUND,
                run=None,
                error=ToolNotFoundError(ToolName.PYTEST, self.tools_config.pytest_bin),
                reason="not_found",
            )
        except ToolExecutionError as exc:
            outcome = _classify_pytest_execution_error(exc)
        else:
            outcome = _classify_pytest_result(result)
        return outcome


def _require_json_report_path(kwargs: Mapping[str, object]) -> Path:
    json_report_path_obj = kwargs.get("json_report_path")
    if not isinstance(json_report_path_obj, Path):
        message = "pytest plugin requires json_report_path of type Path"
        raise TypeError(message)
    return json_report_path_obj


async def _load_pytest_report(json_report_path: Path) -> TestReport:
    def _is_file() -> bool:
        return json_report_path.is_file()

    output_exists = await to_thread.run_sync(_is_file)
    if not output_exists:
        return TestReport.empty()

    def _load_and_parse() -> TestReport:
        try:
            payload = json.loads(json_report_path.read_text(encoding="utf-8"))
            return _parse_pytest_json(payload, json_report_path)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Failed to parse pytest JSON: %s", exc)
            return TestReport.empty()

    return await to_thread.run_sync(_load_and_parse)


def _looks_like_missing_json_report_output(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    if "--json-report" not in combined:
        return False
    return "unrecognized arguments" in combined or "unknown option" in combined


def _is_no_tests_result(result: ToolRunResult) -> bool:
    return result.returncode == _PYTEST_NO_TESTS_EXIT_CODE


def _classify_pytest_execution_error(exc: ToolExecutionError) -> _PytestRunOutcome:
    if _looks_like_missing_json_report_output(exc.result.stdout, exc.result.stderr):
        log.warning("pytest json-report flags unsupported; skipping test ingestion")
        return _PytestRunOutcome(
            status=ToolStatus.SKIPPED,
            run=exc.result,
            error=None,
            reason="missing_json_report",
        )
    if _is_no_tests_result(exc.result):
        return _PytestRunOutcome(
            status=ToolStatus.SKIPPED,
            run=exc.result,
            error=None,
            reason="no_tests",
        )
    return _PytestRunOutcome(
        status=ToolStatus.FAILED,
        run=exc.result,
        error=exc,
        reason="execution_error",
    )


def _classify_pytest_result(result: ToolRunResult) -> _PytestRunOutcome:
    if _is_no_tests_result(result):
        return _PytestRunOutcome(
            status=ToolStatus.SKIPPED,
            run=result,
            error=None,
            reason="no_tests",
        )
    if _looks_like_missing_json_report_output(result.stdout, result.stderr):
        log.warning("pytest json-report flags unsupported; skipping test ingestion")
        return _PytestRunOutcome(
            status=ToolStatus.SKIPPED,
            run=result,
            error=None,
            reason="missing_json_report",
        )
    if result.ok:
        return _PytestRunOutcome(status=ToolStatus.OK, run=result, error=None)
    return _PytestRunOutcome(
        status=ToolStatus.FAILED,
        run=result,
        error=ToolExecutionError(result),
        reason="nonzero_exit",
    )
