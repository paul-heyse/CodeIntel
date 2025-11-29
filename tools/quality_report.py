"""Run the quality gate commands and emit a structured JSON report.

The script executes Ruff, Pyright, Pyrefly, and pytest under `uv run`, captures their
outputs, and serializes a machine-friendly report for downstream agents.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from asyncio.subprocess import PIPE
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict


class CommandResultPayload(TypedDict):
    """Store serialized result for a single command execution."""

    name: str
    command: list[str]
    status: str
    return_code: int
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout: list[str]
    stderr: list[str]
    exception: NotRequired[str]
    details: NotRequired[PytestDetailsPayload]


class PytestSummaryPayload(TypedDict):
    """Store aggregated pytest summary metrics."""

    collected: int
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    duration_seconds: float
    exitcode: int


class PytestFailurePayload(TypedDict, total=False):
    """Store details for a single failing pytest phase."""

    nodeid: str
    phase: str
    outcome: str
    duration_seconds: float
    longrepr: str
    lineno: int


class PytestWarningPayload(TypedDict, total=False):
    """Store pytest warning metadata."""

    message: str
    category: str
    filename: str
    lineno: int
    when: str


class PytestResultsPayload(TypedDict):
    """Store pytest result details derived from json/junit outputs."""

    summary: PytestSummaryPayload
    failures: list[PytestFailurePayload]
    warnings: list[PytestWarningPayload]
    report_paths: dict[str, str]


class CoverageFilePayload(TypedDict):
    """Store coverage details for a single file."""

    path: str
    percent_covered: float
    missing_lines: int
    num_statements: int


class PytestCoveragePayload(TypedDict):
    """Store pytest coverage metadata."""

    totals: dict[str, float | int | str]
    files_missing: list[CoverageFilePayload]
    coverage_json_path: str
    files_missing_limited: NotRequired[bool]


class PytestOutputSplitPayload(TypedDict):
    """Store pytest stdout split into results vs coverage sections."""

    results_stdout: list[str]
    coverage_stdout: list[str]


class PytestDetailsPayload(TypedDict, total=False):
    """Store pytest-specific structured details."""

    results: PytestResultsPayload
    coverage: PytestCoveragePayload
    stdout_split: PytestOutputSplitPayload


class QualityReportPayload(TypedDict):
    """Store the top-level report payload for all quality commands."""

    suite: str
    generated_at: str
    overall_status: str
    duration_seconds: float
    results: list[CommandResultPayload]
    output_path: str


@dataclass(frozen=True)
class CommandSpec:
    """Store command metadata."""

    name: str
    args: list[str]


@dataclass(frozen=True)
class CommandResult:
    """Store the execution result for a command."""

    spec: CommandSpec
    started_at: datetime.datetime
    ended_at: datetime.datetime
    return_code: int
    stdout: str
    stderr: str
    exception: str | None
    details: PytestDetailsPayload | None = None

    def to_payload(self) -> CommandResultPayload:
        """Convert the result into a JSON-ready payload.

        Returns
        -------
        CommandResultPayload
            Serialized payload for the command execution.
        """
        status = "passed" if self.return_code == 0 else "failed"
        payload: CommandResultPayload = {
            "name": self.spec.name,
            "command": self.spec.args,
            "status": status,
            "return_code": self.return_code,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "duration_seconds": (self.ended_at - self.started_at).total_seconds(),
            "stdout": self.stdout.splitlines(),
            "stderr": self.stderr.splitlines(),
        }
        if self.exception is not None:
            payload["exception"] = self.exception
        if self.details is not None:
            payload["details"] = self.details
        return payload


def _now() -> datetime.datetime:
    """Return an aware timestamp in UTC.

    Returns
    -------
    datetime.datetime
        Current UTC timestamp with timezone information.
    """
    return datetime.datetime.now(datetime.UTC)


async def run_command(spec: CommandSpec, workdir: Path) -> CommandResult:
    """Run a command and capture its outputs.

    Parameters
    ----------
    spec
        The command specification to execute.
    workdir
        Working directory used for the command.

    Returns
    -------
    CommandResult
        Captured stdout, stderr, and timing metadata for the command.
    """
    started_at = _now()
    try:
        process = await asyncio.create_subprocess_exec(
            *spec.args,
            cwd=str(workdir),
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
    except OSError as error:
        return CommandResult(
            spec=spec,
            started_at=started_at,
            ended_at=_now(),
            return_code=-1,
            stdout="",
            stderr=str(error),
            exception=str(error),
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return CommandResult(
        spec=spec,
        started_at=started_at,
        ended_at=_now(),
        return_code=process.returncode if process.returncode is not None else 1,
        stdout=stdout,
        stderr=stderr,
        exception=None,
    )


def split_pytest_stdout(stdout: list[str]) -> PytestOutputSplitPayload | None:
    """Split pytest stdout into results vs coverage sections.

    Returns
    -------
    PytestOutputSplitPayload | None
        Separated stdout sections when coverage output is present, otherwise None.
    """
    marker = "================================ tests coverage ================================"
    if marker not in stdout:
        return None
    coverage_index = stdout.index(marker)
    return PytestOutputSplitPayload(
        results_stdout=stdout[:coverage_index],
        coverage_stdout=stdout[coverage_index:],
    )


def _build_pytest_summary(report_data: Mapping[str, Any]) -> PytestSummaryPayload:
    """Build pytest summary payload from json report data.

    Returns
    -------
    PytestSummaryPayload
        Aggregated pytest summary metrics.
    """
    summary_data = report_data.get("summary", {})
    if not isinstance(summary_data, Mapping):
        summary_data = {}
    return PytestSummaryPayload(
        collected=int(summary_data.get("collected", 0)),
        passed=int(summary_data.get("passed", 0)),
        failed=int(summary_data.get("failed", 0)),
        skipped=int(summary_data.get("skipped", 0)),
        xfailed=int(summary_data.get("xfailed", 0)),
        xpassed=int(summary_data.get("xpassed", 0)),
        duration_seconds=float(report_data.get("duration", 0.0)),
        exitcode=int(report_data.get("exitcode", 0)),
    )


def _extract_pytest_failures(report_data: Mapping[str, Any]) -> list[PytestFailurePayload]:
    """Extract failing/errored pytest phases from the json report.

    Returns
    -------
    list[PytestFailurePayload]
        Collection of failing test phases with metadata.
    """
    tests_data = report_data.get("tests", [])
    if not isinstance(tests_data, list):
        return []

    failures: list[PytestFailurePayload] = []
    for test in tests_data:
        lineno = test.get("lineno")
        for phase_name in ("setup", "call", "teardown"):
            phase = test.get(phase_name)
            if not phase:
                continue
            outcome = phase.get("outcome")
            if outcome in {"passed", None}:
                continue
            failure: PytestFailurePayload = {
                "nodeid": test.get("nodeid", ""),
                "phase": phase_name,
                "outcome": outcome,
            }
            if lineno is not None:
                failure["lineno"] = int(lineno)
            if "duration" in phase:
                failure["duration_seconds"] = float(phase["duration"])
            longrepr = phase.get("longrepr")
            if longrepr:
                failure["longrepr"] = str(longrepr)
            failures.append(failure)
    return failures


def _extract_pytest_warnings(report_data: Mapping[str, Any]) -> list[PytestWarningPayload]:
    """Extract warnings from the pytest json report.

    Returns
    -------
    list[PytestWarningPayload]
        Collection of captured warnings.
    """
    warnings_data = report_data.get("warnings", [])
    if not isinstance(warnings_data, list):
        return []

    warnings_payload: list[PytestWarningPayload] = []
    for warning in warnings_data:
        warning_payload: PytestWarningPayload = {
            "message": warning.get("message", ""),
            "category": warning.get("category", ""),
        }
        if warning.get("filename"):
            warning_payload["filename"] = warning["filename"]
        if warning.get("lineno") is not None:
            warning_payload["lineno"] = int(warning["lineno"])
        if warning.get("when"):
            warning_payload["when"] = warning["when"]
        warnings_payload.append(warning_payload)
    return warnings_payload


def parse_pytest_report(
    report_path: Path, junit_path: Path | None = None
) -> PytestResultsPayload | None:
    """Parse pytest json/junit outputs into structured details.

    Returns
    -------
    PytestResultsPayload | None
        Structured details when the json report is present, otherwise None.
    """
    if not report_path.exists():
        return None
    try:
        report_data: Mapping[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    summary = _build_pytest_summary(report_data)
    failures = _extract_pytest_failures(report_data)
    warnings_payload = _extract_pytest_warnings(report_data)
    report_paths = {"pytest_report": str(report_path)}
    if junit_path is not None and junit_path.exists():
        report_paths["junit_xml"] = str(junit_path)

    return PytestResultsPayload(
        summary=summary,
        failures=failures,
        warnings=warnings_payload,
        report_paths=report_paths,
    )


def _collect_missing_files(
    files_data: Mapping[str, Any], max_files: int
) -> tuple[list[CoverageFilePayload], bool]:
    """
    Extract CoverageFilePayload entries for files with missing lines.

    Returns
    -------
    tuple[list[CoverageFilePayload], bool]
        Missing-file entries sorted by coverage and whether the list was truncated.
    """
    missing_files: list[CoverageFilePayload] = []
    for file_path, file_info in files_data.items():
        if not isinstance(file_info, Mapping):
            continue
        summary = file_info.get("summary")
        if not isinstance(summary, Mapping) or summary is None:
            continue
        missing = int(summary.get("missing_lines", 0))
        if missing == 0:
            continue
        missing_files.append(
            CoverageFilePayload(
                path=file_path,
                percent_covered=float(summary.get("percent_covered", 0.0)),
                missing_lines=missing,
                num_statements=int(summary.get("num_statements", 0)),
            )
        )
    missing_files.sort(key=lambda row: row["percent_covered"])
    limited = len(missing_files) > max_files
    if limited:
        missing_files = missing_files[:max_files]
    return missing_files, limited


def parse_coverage_json(
    coverage_json_path: Path, *, max_files: int = 200
) -> PytestCoveragePayload | None:
    """Parse coverage json for totals and low-coverage files.

    Returns
    -------
    PytestCoveragePayload | None
        Structured coverage data when the json file exists, otherwise None.
    """
    if not coverage_json_path.exists():
        return None
    try:
        coverage_data: Mapping[str, Any] = json.loads(
            coverage_json_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return None

    totals = coverage_data.get("totals", {})
    files_data = coverage_data.get("files", {})
    if not isinstance(files_data, Mapping):
        return None
    files_missing, files_missing_limited = _collect_missing_files(files_data, max_files)

    coverage_payload: PytestCoveragePayload = {
        "totals": totals,
        "files_missing": files_missing,
        "coverage_json_path": str(coverage_json_path),
    }
    if files_missing_limited:
        coverage_payload["files_missing_limited"] = True
    return coverage_payload


def build_pytest_details(result: CommandResult, repo_root: Path) -> PytestDetailsPayload | None:
    """Assemble pytest-specific structured details.

    Returns
    -------
    PytestDetailsPayload | None
        Structured pytest metadata when report artifacts are available.
    """
    pytest_report_path = repo_root / "build/test-results/pytest-report.json"
    junit_path = repo_root / "build/test-results/junit.xml"
    coverage_json_path = repo_root / "build/coverage/coverage.json"

    details: PytestDetailsPayload = {}

    split_output = split_pytest_stdout(result.stdout.splitlines())
    if split_output is not None:
        details["stdout_split"] = split_output

    pytest_results = parse_pytest_report(pytest_report_path, junit_path=junit_path)
    if pytest_results is not None:
        details["results"] = pytest_results

    coverage_details = parse_coverage_json(coverage_json_path)
    if coverage_details is not None:
        details["coverage"] = coverage_details

    if details:
        return details
    return None


def generate_report(results: list[CommandResult], output_path: Path) -> QualityReportPayload:
    """Build the aggregate quality report payload.

    Parameters
    ----------
    results
        Collection of command results to summarize.
    output_path
        Destination path used when persisting the report to disk.

    Returns
    -------
    QualityReportPayload
        Structured payload for all commands, including timing metadata.
    """
    overall_status = "passed"
    if any(result.return_code != 0 for result in results):
        overall_status = "failed"

    if not results:
        duration_seconds = 0.0
    else:
        start = min(result.started_at for result in results)
        end = max(result.ended_at for result in results)
        duration_seconds = (end - start).total_seconds()

    return QualityReportPayload(
        suite="quality_checks",
        generated_at=_now().isoformat(),
        overall_status=overall_status,
        duration_seconds=duration_seconds,
        results=[result.to_payload() for result in results],
        output_path=str(output_path),
    )


def parse_args() -> Path:
    """Parse CLI arguments.

    Returns
    -------
    Path
        Destination path for the JSON report.
    """
    parser = argparse.ArgumentParser(
        description="Run quality checks and emit a machine-readable report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/quality-results/quality_report.json"),
        help=("Path for the JSON report (default: build/quality-results/quality_report.json)"),
    )
    args = parser.parse_args()
    return args.output


async def run_suite(commands: list[CommandSpec], repo_root: Path) -> list[CommandResult]:
    """Execute each command sequentially and capture results.

    Parameters
    ----------
    commands
        Collection of commands to execute.
    repo_root
        Working directory applied to each command.

    Returns
    -------
    list[CommandResult]
        Captured results for each command.
    """
    return [await run_command(command, repo_root) for command in commands]


def main() -> int:
    """Execute the quality suite and persist the report.

    Returns
    -------
    int
        Zero when all commands succeed, otherwise a non-zero exit status.
    """
    output_path = parse_args().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "pyproject.toml").exists():
        sys.stderr.write("Error: run this script from the repository context.\n")
        return 1

    commands = [
        CommandSpec(name="ruff_check", args=["uv", "run", "ruff", "check", "--fix"]),
        CommandSpec(
            name="pyright",
            args=["uv", "run", "pyright", "--warnings", "--pythonversion=3.13"],
        ),
        CommandSpec(name="pyrefly", args=["uv", "run", "pyrefly", "check"]),
        CommandSpec(name="pytest", args=["uv", "run", "pytest", "-q"]),
    ]

    results = asyncio.run(run_suite(commands, repo_root))
    enriched_results: list[CommandResult] = []
    for command_result in results:
        updated_result = command_result
        if command_result.spec.name == "pytest":
            pytest_details = build_pytest_details(command_result, repo_root)
            if pytest_details is not None:
                updated_result = replace(command_result, details=pytest_details)
        enriched_results.append(updated_result)

    report = generate_report(enriched_results, output_path)
    serialized = json.dumps(report, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{serialized}\n", encoding="utf-8")
    sys.stdout.write(f"{serialized}\n")

    if report["overall_status"] == "passed":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
