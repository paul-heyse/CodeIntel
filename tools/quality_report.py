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
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict


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
        default=Path("document_output/quality_report.json"),
        help=("Path for the JSON report (default: document_output/quality_report.json)"),
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

    report = generate_report(results, output_path)
    serialized = json.dumps(report, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{serialized}\n", encoding="utf-8")
    sys.stdout.write(f"{serialized}\n")

    if report["overall_status"] == "passed":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
