"""Tests for build providers and parsing helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.providers import (
    RealCoverageCollector,
    RealGitHistoryProvider,
    RealScipIndexer,
    RealTestReporter,
    RealTypeChecker,
    SubprocessToolRunner,
)
from codeintel.build.types import ToolRunResult as BuildToolRunResult
from codeintel.config.models import ToolsConfig
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true
from tests._helpers.fakes.tools import ToolRunOptions, make_tool_run_result

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from codeintel.ingestion.engine.infrastructure.runner import (
        ToolRunResult as IngestionToolRunResult,
    )

pytestmark = pytest.mark.anyio


def _to_build_result(
    result: BuildToolRunResult | IngestionToolRunResult,
) -> BuildToolRunResult:
    """Normalize ingestion ToolRunResult to the build protocol shape.

    Returns
    -------
    BuildToolRunResult
        Tool run result compatible with build providers.
    """
    if isinstance(result, BuildToolRunResult):
        return result
    tool_name = result.tool.value if hasattr(result.tool, "value") else str(result.tool)
    return BuildToolRunResult(
        tool=tool_name,
        args=tuple(result.args),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_ms=int(result.duration_s * 1000),
    )


class StubToolRunner(SubprocessToolRunner):
    """SubprocessToolRunner subclass that returns a fixed result."""

    def __init__(
        self,
        result: BuildToolRunResult | IngestionToolRunResult,
        *,
        hook: Callable[[str, Sequence[str], Path, int | None, Mapping[str, str] | None], None]
        | None = None,
    ) -> None:
        """Initialize stub runner with a result and optional hook."""
        super().__init__(ToolsConfig.default())
        self.result = _to_build_result(result)
        self.hook = hook
        self.calls: list[tuple[str, tuple[str, ...], Path]] = []

    async def run(
        self,
        tool: str,
        args: Sequence[str],
        cwd: Path,
        *,
        timeout_ms: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> BuildToolRunResult:
        """Return the configured result after recording the call.

        Returns
        -------
        ToolRunResult
            Preconfigured result for the call.
        """
        self.calls.append((tool, tuple(args), cwd))
        if self.hook is not None:
            self.hook(tool, args, cwd, timeout_ms, env)
        return self.result


class _FakeProcess:
    """Minimal asyncio subprocess stand-in."""

    def __init__(
        self,
        stdout: bytes = b"ok",
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return configured stdout/stderr.

        Returns
        -------
        tuple[bytes, bytes]
            Captured stdout and stderr.
        """
        return (self.stdout, self.stderr)


class _TimeoutProcess(_FakeProcess):
    """Process whose communicate raises TimeoutError."""

    async def communicate(self) -> tuple[bytes, bytes]:
        """Raise TimeoutError to simulate communication timeout.

        Raises
        ------
        TimeoutError
            Always raised to simulate a timeout.
        """
        self.returncode = -1
        raise TimeoutError


@pytest.mark.anyio
async def test_subprocess_runner_success(tmp_path: Path) -> None:
    """SubprocessToolRunner returns decoded output on success."""

    async def make_process(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        _ = (args, kwargs)
        await asyncio.sleep(0)
        return cast("asyncio.subprocess.Process", _FakeProcess())

    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        subprocess_runner=make_process,
    )

    result = await runner.run("pyright", ["--version"], tmp_path)

    expect_true(result.success)
    expect_equal(result.stdout, "ok")
    expect_false(bool(result.stderr))


@pytest.mark.anyio
async def test_subprocess_runner_timeout(tmp_path: Path) -> None:
    """Timeouts yield returncode -1 and timeout message."""

    async def make_timeout_process(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        _ = (args, kwargs)
        await asyncio.sleep(0)
        return cast("asyncio.subprocess.Process", _TimeoutProcess())

    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        subprocess_runner=make_timeout_process,
    )

    result = await runner.run("pyright", [], tmp_path)

    expect_equal(result.returncode, -1)
    expect_in("Timeout after", result.stderr)


@pytest.mark.anyio
async def test_subprocess_runner_missing_binary(tmp_path: Path) -> None:
    """Missing executable returns tool-not-found error."""

    async def raise_missing(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        _ = (args, kwargs)
        message = "missing"
        await asyncio.sleep(0)
        raise FileNotFoundError(message)

    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        subprocess_runner=raise_missing,
    )

    result = await runner.run("pyright", [], tmp_path)

    expect_equal(result.returncode, -1)
    expect_in("Tool not found", result.stderr)


@pytest.mark.anyio
async def test_subprocess_runner_unexpected_error(tmp_path: Path) -> None:
    """Unexpected exceptions are captured in the result."""

    async def raise_error(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        _ = (args, kwargs)
        message = "boom"
        await asyncio.sleep(0)
        raise RuntimeError(message)

    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        subprocess_runner=raise_error,
    )

    result = await runner.run("pyright", [], tmp_path)

    expect_equal(result.returncode, -1)
    expect_equal(result.stderr, "boom")


def test_subprocess_runner_is_available() -> None:
    """is_available reflects shutil.which."""
    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        which_resolver=lambda _path: "/bin/tool",
    )
    expect_true(runner.is_available("git"))
    runner = SubprocessToolRunner(
        ToolsConfig.default(),
        which_resolver=lambda _path: None,
    )
    expect_false(runner.is_available("git"))


@pytest.mark.anyio
async def test_scip_indexer_index_success(tmp_path: Path) -> None:
    """Index reports success when output exists."""
    output_path = tmp_path / "index.scip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("content", encoding="utf-8")
    runner = StubToolRunner(
        make_tool_run_result("scip-python", options=ToolRunOptions(returncode=0, duration_s=0.005))
    )
    indexer = RealScipIndexer(runner)

    result = await indexer.index(
        tmp_path,
        output_path,
        include_patterns=("*.py",),
        exclude_patterns=("*.txt",),
    )

    expect_true(result.success)
    tool, args, call_cwd = runner.calls[0]
    expect_equal(tool, "scip-python")
    expect_in("--include", args)
    expect_in("--exclude", args)
    expect_equal(call_cwd, tmp_path)


@pytest.mark.anyio
async def test_scip_indexer_index_failure(tmp_path: Path) -> None:
    """Index returns error when tool fails."""
    output_path = tmp_path / "index.scip"
    runner = StubToolRunner(
        make_tool_run_result(
            "scip-python",
            options=ToolRunOptions(returncode=1, stderr="index failed", duration_s=0.001),
        )
    )
    indexer = RealScipIndexer(runner)

    result = await indexer.index(tmp_path, output_path)

    expect_false(result.success)
    expect_in("index failed", (result.error_message or ""))


@pytest.mark.anyio
async def test_scip_parse_success(tmp_path: Path) -> None:
    """Parse extracts symbols and occurrences from JSON."""
    scip_path = tmp_path / "index.scip"
    json_path = tmp_path / "index.json"
    data = {
        "documents": [
            {
                "relative_path": "file.py",
                "symbols": [
                    {
                        "symbol": "sym1",
                        "display_name": "Sym1",
                        "kind": "function",
                        "documentation": "doc",
                        "signature_documentation": "sig",
                    }
                ],
                "occurrences": [
                    {
                        "symbol": "sym1",
                        "range": [1, 2, 3, 4],
                        "symbol_roles": "definition",
                    }
                ],
            }
        ]
    }
    runner = StubToolRunner(
        make_tool_run_result(
            "scip", options=ToolRunOptions(returncode=0, stdout=json.dumps(data), duration_s=0.001)
        )
    )
    indexer = RealScipIndexer(runner)

    result = await indexer.parse(scip_path, json_path)

    expect_true(result.success)
    expect_equal(result.json_path, json_path)
    expect_equal(result.symbols[0].symbol, "sym1")
    expect_equal(result.occurrences[0].path, "file.py")


@pytest.mark.anyio
async def test_scip_parse_handles_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON surfaces as an error."""
    scip_path = tmp_path / "index.scip"
    json_path = tmp_path / "index.json"
    runner = StubToolRunner(
        make_tool_run_result(
            "scip",
            options=ToolRunOptions(returncode=0, stdout="not json", stderr="bad", duration_s=0.001),
        )
    )
    indexer = RealScipIndexer(runner)

    result = await indexer.parse(scip_path, json_path)

    expect_false(result.success)
    expect_in("Failed to process", (result.error_message or ""))


@pytest.mark.anyio
async def test_scip_parse_handles_failure(tmp_path: Path) -> None:
    """Non-zero tool exit returns failure result."""
    scip_path = tmp_path / "index.scip"
    json_path = tmp_path / "index.json"
    runner = StubToolRunner(
        make_tool_run_result(
            "scip", options=ToolRunOptions(returncode=1, stderr="parse failed", duration_s=0.001)
        )
    )
    indexer = RealScipIndexer(runner)

    result = await indexer.parse(scip_path, json_path)

    expect_false(result.success)
    expect_true((result.error_message or "").startswith("parse failed"))


@pytest.mark.anyio
async def test_type_checker_parses_diagnostics(tmp_path: Path) -> None:
    """Type checker parses diagnostics and counts warnings."""
    data = {
        "generalDiagnostics": [
            {
                "file": "module.py",
                "severity": "warning",
                "rule": "warn",
                "message": "be careful",
                "range": {"start": {"line": 1, "character": 2}},
            }
        ]
    }
    runner = StubToolRunner(
        make_tool_run_result(
            "pyright",
            options=ToolRunOptions(returncode=0, stdout=json.dumps(data), duration_s=0.001),
        )
    )
    checker = RealTypeChecker(runner)

    result = await checker.check(tmp_path, paths=[tmp_path / "module.py"])

    expect_true(result.success)
    expect_equal(result.warning_count, 1)
    expect_equal(result.error_count, 0)


@pytest.mark.anyio
async def test_type_checker_handles_parse_failure(tmp_path: Path) -> None:
    """Invalid JSON produces a parse_error diagnostic."""
    runner = StubToolRunner(
        make_tool_run_result(
            "pyright",
            options=ToolRunOptions(returncode=0, stdout="not json", stderr="bad", duration_s=0.001),
        )
    )
    checker = RealTypeChecker(runner)

    result = await checker.check(tmp_path)

    expect_false(result.success)
    expect_equal(result.error_count, 1)
    expect_equal(result.diagnostics[0].code, "parse_error")


@pytest.mark.anyio
async def test_coverage_collector_parses_json(tmp_path: Path) -> None:
    """Coverage collector parses coverage json output."""
    coverage_file = tmp_path / ".coverage"
    json_path = coverage_file.with_suffix(".json")
    json_data = {
        "files": {
            "module.py": {
                "executed_lines": [1, 2],
                "missing_lines": [3],
                "excluded_lines": [],
            }
        }
    }

    def write_json(
        tool: str,
        args: Sequence[str],
        cwd: Path,
        timeout_ms: int | None,
        env: Mapping[str, str] | None,
    ) -> None:
        _ = (tool, args, cwd, timeout_ms, env)
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

    runner = StubToolRunner(make_tool_run_result("coverage"), hook=write_json)
    collector = RealCoverageCollector(runner)

    result = await collector.collect(coverage_file)

    expect_in("module.py", result)
    coverage = result["module.py"]
    expect_equal(coverage.covered_lines, frozenset({1, 2}))
    expect_equal(coverage.missing_lines, frozenset({3}))


@pytest.mark.anyio
async def test_coverage_collector_missing_file(tmp_path: Path) -> None:
    """Missing coverage json yields empty result."""
    coverage_file = tmp_path / ".coverage"

    runner = StubToolRunner(make_tool_run_result("coverage"))
    collector = RealCoverageCollector(runner)

    result = await collector.collect(coverage_file)

    expect_equal(result, {})


@pytest.mark.anyio
async def test_test_reporter_collects_results(tmp_path: Path) -> None:
    """Parse pytest JSON report into TestResult records."""
    report_path = tmp_path / "report.json"
    data = {
        "tests": [
            {
                "nodeid": "tests/test_sample.py::test_case",
                "outcome": "passed",
                "duration": 0.02,
                "keywords": {"slow": True},
            }
        ]
    }
    report_path.write_text(json.dumps(data), encoding="utf-8")
    reporter = RealTestReporter()

    results = await reporter.collect(report_path)

    expect_equal(len(results), 1)
    test = results[0]
    expect_equal(test.node_id, "tests/test_sample.py::test_case")
    expect_equal(test.outcome, "passed")
    expect_equal(test.duration_ms, 20)


@pytest.mark.anyio
async def test_test_reporter_missing_file(tmp_path: Path) -> None:
    """Missing pytest report yields empty tuple."""
    report_path = tmp_path / "report.json"
    reporter = RealTestReporter()

    results = await reporter.collect(report_path)

    expect_equal(results, ())


@pytest.mark.anyio
async def test_git_history_log_parses_output(tmp_path: Path) -> None:
    """Git log parsing extracts stats and metadata."""
    sha = "a" * 40
    output = "\n".join(
        [
            f"{sha}|Alice|alice@example.com|2024-01-01T00:00:00Z|Initial commit",
            " 1 file changed, 2 insertions(+), 3 deletions(-)",
        ]
    )
    runner = StubToolRunner(make_tool_run_result("git", options=ToolRunOptions(stdout=output)))
    provider = RealGitHistoryProvider(runner)

    entries = await provider.log(tmp_path)

    expect_equal(len(entries), 1)
    entry = entries[0]
    expect_equal(entry.files_changed, 1)
    expect_equal(entry.insertions, 2)
    expect_equal(entry.deletions, 3)


@pytest.mark.anyio
async def test_git_history_log_failure(tmp_path: Path) -> None:
    """Non-zero git log result returns empty tuple."""
    runner = StubToolRunner(
        make_tool_run_result("git", options=ToolRunOptions(returncode=1, stderr="bad"))
    )
    provider = RealGitHistoryProvider(runner)

    entries = await provider.log(tmp_path)

    expect_equal(entries, ())


@pytest.mark.anyio
async def test_git_history_blame_parses_output(tmp_path: Path) -> None:
    """Git blame output is parsed into a mapping."""
    sha = "b" * 40
    output = "\n".join(
        [
            f"{sha} 1 1 1",
            "author Bob",
            "author-mail <bob@example.com>",
            "author-time 1700000000",
            "summary Something",
            "\tprint('hi')",
        ]
    )
    runner = StubToolRunner(make_tool_run_result("git", options=ToolRunOptions(stdout=output)))
    provider = RealGitHistoryProvider(runner)

    mapping = await provider.blame(tmp_path, Path("file.py"))

    expect_in(1, mapping)
    expect_equal(mapping[1].author, "Bob")
