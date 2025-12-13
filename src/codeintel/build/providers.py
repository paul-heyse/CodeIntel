"""Production implementations of DI protocols.

This module provides concrete implementations of the protocols
defined in protocols.py. These implementations use actual system
resources (subprocess, filesystem, etc.).

The factory function create_default_providers() creates a complete
set of providers wired together.

Example
-------
>>> from codeintel.build.providers import create_default_providers
>>> from codeintel.config.models import ToolsConfig
>>> providers = create_default_providers(ToolsConfig.default())
>>> result = await providers.tool_runner.run("pyright", ["--version"], Path("."))
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.types import (
    CoverageData,
    GitLogEntry,
    ScipIndexResult,
    ScipOccurrence,
    ScipParseResult,
    ScipSymbol,
    TestResult,
    ToolRunResult,
    TypeCheckResult,
    TypeDiagnostic,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from codeintel.config.models import ToolsConfig

log = logging.getLogger(__name__)

__all__ = [
    "Providers",
    "RealCoverageCollector",
    "RealGitHistoryProvider",
    "RealScipIndexer",
    "RealTestReporter",
    "RealTypeChecker",
    "SubprocessToolRunner",
    "create_default_providers",
]


_SCIP_SYMBOL_MIN_PARTS = 4


_GIT_LOG_RECORD_PARTS = 5


_GIT_COMMIT_HASH_LENGTH = 40


def _parse_git_stat_line(lines: list[str], current_idx: int) -> tuple[bool, int, int, int]:
    """Parse git shortstat line to extract file changes.

    Parameters
    ----------
    lines
        All output lines.
    current_idx
        Current index in lines.

    Returns
    -------
    tuple[bool, int, int, int]
        (has_stat, files_changed, insertions, deletions).
    """
    if current_idx + 1 >= len(lines):
        return (False, 0, 0, 0)

    stat_line = lines[current_idx + 1].strip()
    if "file" not in stat_line:
        return (False, 0, 0, 0)

    files_match = re.search(r"(\d+) files? changed", stat_line)
    ins_match = re.search(r"(\d+) insertions?", stat_line)
    del_match = re.search(r"(\d+) deletions?", stat_line)

    return (
        True,
        int(files_match.group(1)) if files_match else 0,
        int(ins_match.group(1)) if ins_match else 0,
        int(del_match.group(1)) if del_match else 0,
    )


@dataclass
class _BlameParseState:
    """Internal state for blame parsing."""

    current_sha: str = ""
    current_line: int = 0
    author: str = ""
    author_email: str = ""
    author_time: str = ""
    summary: str = ""


def _parse_blame_output(lines: list[str]) -> dict[int, GitLogEntry]:
    """Parse git blame porcelain output lines.

    Parameters
    ----------
    lines
        Blame output lines.

    Returns
    -------
    dict[int, GitLogEntry]
        Line to commit mapping.
    """
    result: dict[int, GitLogEntry] = {}
    commit_cache: dict[str, GitLogEntry] = {}
    state = _BlameParseState()

    for line in lines:
        if line.startswith("\t"):
            _record_blame_line(state, result, commit_cache)
        elif " " in line:
            _parse_blame_field(line, state)

    return result


def _record_blame_line(
    state: _BlameParseState,
    result: dict[int, GitLogEntry],
    commit_cache: dict[str, GitLogEntry],
) -> None:
    """Record a blame line mapping."""
    if not state.current_sha or not state.current_line:
        return

    if state.current_sha not in commit_cache:
        commit_cache[state.current_sha] = GitLogEntry(
            sha=state.current_sha,
            author=state.author,
            author_email=state.author_email,
            date=state.author_time,
            message=state.summary,
        )
    result[state.current_line] = commit_cache[state.current_sha]


def _parse_blame_field(line: str, state: _BlameParseState) -> None:
    """Parse a single blame field line."""
    parts = line.split(" ", 1)
    key = parts[0]
    value = parts[1] if len(parts) > 1 else ""

    if len(key) == _GIT_COMMIT_HASH_LENGTH:
        state.current_sha = key
        line_parts = value.split()
        state.current_line = int(line_parts[1]) if len(line_parts) > 1 else 0
    elif key == "author":
        state.author = value
    elif key == "author-mail":
        state.author_email = value.strip("<>")
    elif key == "author-time":
        state.author_time = value
    elif key == "summary":
        state.summary = value


@dataclass
class SubprocessToolRunner:
    """Production tool runner using asyncio subprocess.

    This implementation resolves tool paths from ToolsConfig and
    executes them using asyncio.create_subprocess_exec.

    Attributes
    ----------
    tools_config
        Configuration with tool binary paths.
    default_timeout_ms
        Default timeout if not specified in run().
    subprocess_runner
        Callable used to create subprocesses (injectable for testing).
    which_resolver
        Callable used to resolve executable paths (injectable for testing).
    """

    tools_config: ToolsConfig
    default_timeout_ms: int = 60000
    subprocess_runner: Callable[..., Awaitable[asyncio.subprocess.Process]] = (
        asyncio.create_subprocess_exec
    )
    which_resolver: Callable[[str], str | None] = shutil.which

    def _resolve_tool_path(self, tool: str) -> str:
        """Resolve tool name to executable path.

        Parameters
        ----------
        tool
            Tool name (e.g., "scip-python", "pyright").

        Returns
        -------
        str
            Path to executable.
        """
        tool_map: dict[str, str] = {
            "scip-python": self.tools_config.scip_python_bin,
            "scip": self.tools_config.scip_bin,
            "pyright": self.tools_config.pyright_bin,
            "pyrefly": self.tools_config.pyrefly_bin,
            "ruff": self.tools_config.ruff_bin,
            "coverage": self.tools_config.coverage_bin,
            "pytest": self.tools_config.pytest_bin,
            "git": self.tools_config.git_bin,
        }
        return tool_map.get(tool, tool)

    async def run(
        self,
        tool: str,
        args: Sequence[str],
        cwd: Path,
        *,
        timeout_ms: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ToolRunResult:
        """Run an external tool via subprocess.

        Parameters
        ----------
        tool
            Tool name to run.
        args
            Command-line arguments.
        cwd
            Working directory.
        timeout_ms
            Timeout in milliseconds.
        env
            Additional environment variables.

        Returns
        -------
        ToolRunResult
            Result with exit code and output.
        """
        tool_path = self._resolve_tool_path(tool)
        timeout = (timeout_ms or self.default_timeout_ms) / 1000.0

        run_env = self.tools_config.build_env(tool)
        if env:
            run_env.update(env)

        start_time = time.monotonic()

        try:
            log.debug("Running %s %s in %s", tool_path, args, cwd)

            proc = await self.subprocess_runner(
                tool_path,
                *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=run_env,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            duration_ms = int((time.monotonic() - start_time) * 1000)

            return ToolRunResult(
                tool=tool,
                args=tuple(args),
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
            )

        except TimeoutError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ToolRunResult(
                tool=tool,
                args=tuple(args),
                returncode=-1,
                stderr=f"Timeout after {duration_ms}ms",
                duration_ms=duration_ms,
            )

        except FileNotFoundError:
            return ToolRunResult(
                tool=tool,
                args=tuple(args),
                returncode=-1,
                stderr=f"Tool not found: {tool_path}",
            )

        except Exception as e:
            log.exception("Error running %s", tool)
            return ToolRunResult(
                tool=tool,
                args=tuple(args),
                returncode=-1,
                stderr=str(e),
            )

    def is_available(self, tool: str) -> bool:
        """Check if a tool is available.

        Parameters
        ----------
        tool
            Tool name to check.

        Returns
        -------
        bool
            True if executable exists.
        """
        tool_path = self._resolve_tool_path(tool)
        return self.which_resolver(tool_path) is not None


@dataclass
class RealScipIndexer:
    """Production SCIP indexer using scip-python and scip tools.

    Attributes
    ----------
    tool_runner
        Tool runner for subprocess execution.
    """

    tool_runner: SubprocessToolRunner

    async def index(
        self,
        repo_root: Path,
        output_path: Path,
        *,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> ScipIndexResult:
        """Generate SCIP index using scip-python.

        Parameters
        ----------
        repo_root
            Repository root directory.
        output_path
            Path for output index.scip file.
        include_patterns
            Optional glob patterns to include.
        exclude_patterns
            Optional glob patterns to exclude.

        Returns
        -------
        ScipIndexResult
            Result with success status.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = ["index", str(repo_root), "--output", str(output_path)]

        if include_patterns:
            for pattern in include_patterns:
                args.extend(["--include", pattern])
        if exclude_patterns:
            for pattern in exclude_patterns:
                args.extend(["--exclude", pattern])

        result = await self.tool_runner.run(
            "scip-python",
            args,
            repo_root,
            timeout_ms=300000,
        )

        path_exists = await asyncio.to_thread(output_path.exists)
        if result.success and path_exists:
            return ScipIndexResult(
                success=True,
                index_path=output_path,
                duration_ms=result.duration_ms,
            )

        return ScipIndexResult(
            success=False,
            error_message=result.stderr or "SCIP indexing failed",
            duration_ms=result.duration_ms,
        )

    async def parse(
        self,
        scip_path: Path,
        output_json_path: Path,
    ) -> ScipParseResult:
        """Parse SCIP index to JSON using scip tool.

        Parameters
        ----------
        scip_path
            Path to index.scip file.
        output_json_path
            Path for output JSON file.

        Returns
        -------
        ScipParseResult
            Result with extracted symbols and occurrences.
        """
        await asyncio.to_thread(output_json_path.parent.mkdir, parents=True, exist_ok=True)

        result = await self.tool_runner.run(
            "scip",
            ["print", "--json", str(scip_path)],
            scip_path.parent,
            timeout_ms=60000,
        )

        if not result.success:
            return ScipParseResult(
                success=False,
                error_message=result.stderr or "SCIP parsing failed",
            )

        try:
            await asyncio.to_thread(output_json_path.write_text, result.stdout, encoding="utf-8")

            data = json.loads(result.stdout)
            symbols = self._extract_symbols(data)
            occurrences = self._extract_occurrences(data)

            return ScipParseResult(
                success=True,
                symbols=symbols,
                occurrences=occurrences,
                json_path=output_json_path,
            )
        except (json.JSONDecodeError, OSError) as e:
            return ScipParseResult(
                success=False,
                error_message=f"Failed to process SCIP JSON: {e}",
            )

    @staticmethod
    def _extract_symbols(data: dict[str, object]) -> tuple[ScipSymbol, ...]:
        """Extract symbols from SCIP JSON.

        Parameters
        ----------
        data
            Parsed SCIP JSON.

        Returns
        -------
        tuple[ScipSymbol, ...]
            Extracted symbols.
        """
        symbols: list[ScipSymbol] = []

        documents = data.get("documents", [])
        if not isinstance(documents, list):
            return ()
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            for sym in doc.get("symbols", []):
                if not isinstance(sym, dict):
                    continue
                symbols.append(
                    ScipSymbol(
                        symbol=str(sym.get("symbol", "")),
                        name=str(sym.get("display_name", sym.get("symbol", ""))),
                        kind=str(sym.get("kind", "unknown")),
                        documentation=sym.get("documentation"),
                        signature=sym.get("signature_documentation"),
                    )
                )
        return tuple(symbols)

    @staticmethod
    def _extract_occurrences(data: dict[str, object]) -> tuple[ScipOccurrence, ...]:
        """Extract occurrences from SCIP JSON.

        Parameters
        ----------
        data
            Parsed SCIP JSON.

        Returns
        -------
        tuple[ScipOccurrence, ...]
            Extracted occurrences.
        """
        occurrences: list[ScipOccurrence] = []
        default_range = [0, 0, 0, 0]
        documents = data.get("documents", [])
        if not isinstance(documents, list):
            return ()
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            path = str(doc.get("relative_path", ""))
            for occ in doc.get("occurrences", []):
                if not isinstance(occ, dict):
                    continue
                range_data = occ.get("range", default_range)
                if not isinstance(range_data, list) or len(range_data) < _SCIP_SYMBOL_MIN_PARTS:
                    range_data = default_range
                occurrences.append(
                    ScipOccurrence(
                        symbol=str(occ.get("symbol", "")),
                        path=path,
                        line=int(range_data[0]),
                        character=int(range_data[1]),
                        end_line=int(range_data[2]),
                        end_character=int(range_data[3]),
                        role=str(occ.get("symbol_roles", "reference")),
                    )
                )
        return tuple(occurrences)


@dataclass
class RealTypeChecker:
    """Production type checker using pyright.

    Attributes
    ----------
    tool_runner
        Tool runner for subprocess execution.
    """

    tool_runner: SubprocessToolRunner

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> TypeCheckResult:
        """Run pyright type checking.

        Parameters
        ----------
        repo_root
            Repository root directory.
        paths
            Optional specific paths to check.
        config_path
            Optional path to pyrightconfig.json.

        Returns
        -------
        TypeCheckResult
            Result with diagnostics.
        """
        args = ["--outputjson"]

        if config_path:
            args.extend(["--project", str(config_path)])

        if paths:
            args.extend(str(p) for p in paths)

        result = await self.tool_runner.run(
            "pyright",
            args,
            repo_root,
            timeout_ms=180000,
        )

        try:
            data = json.loads(result.stdout) if result.stdout else {}
            diagnostics = self._parse_diagnostics(data)
            error_count = sum(1 for d in diagnostics if d.severity == "error")
            warning_count = sum(1 for d in diagnostics if d.severity == "warning")

            return TypeCheckResult(
                success=error_count == 0,
                diagnostics=diagnostics,
                error_count=error_count,
                warning_count=warning_count,
                duration_ms=result.duration_ms,
            )
        except json.JSONDecodeError:
            return TypeCheckResult(
                success=False,
                diagnostics=(
                    TypeDiagnostic(
                        path="",
                        line=0,
                        character=0,
                        severity="error",
                        code="parse_error",
                        message=f"Failed to parse pyright output: {result.stderr}",
                        source="pyright",
                    ),
                ),
                error_count=1,
                duration_ms=result.duration_ms,
            )

    @staticmethod
    def _parse_diagnostics(
        data: dict[str, object],
    ) -> tuple[TypeDiagnostic, ...]:
        """Parse pyright JSON output into diagnostics.

        Parameters
        ----------
        data
            Parsed pyright JSON output.

        Returns
        -------
        tuple[TypeDiagnostic, ...]
            Extracted diagnostics.
        """
        diagnostics: list[TypeDiagnostic] = []
        general_diagnostics = data.get("generalDiagnostics", [])
        if not isinstance(general_diagnostics, list):
            return ()

        for diag in general_diagnostics:
            if not isinstance(diag, dict):
                continue
            range_data = diag.get("range", {})
            if not isinstance(range_data, dict):
                range_data = {}
            start = range_data.get("start", {})
            if not isinstance(start, dict):
                start = {}

            diagnostics.append(
                TypeDiagnostic(
                    path=str(diag.get("file", "")),
                    line=int(start.get("line", 0)),
                    character=int(start.get("character", 0)),
                    severity=str(diag.get("severity", "error")),
                    code=str(diag.get("rule", "")),
                    message=str(diag.get("message", "")),
                    source="pyright",
                )
            )
        return tuple(diagnostics)


@dataclass
class RealCoverageCollector:
    """Production coverage collector using coverage.py.

    Attributes
    ----------
    tool_runner
        Tool runner for subprocess execution.
    """

    tool_runner: SubprocessToolRunner

    async def collect(
        self,
        coverage_file: Path,
    ) -> Mapping[str, CoverageData]:
        """Collect coverage data from a coverage file.

        Parameters
        ----------
        coverage_file
            Path to coverage data file.

        Returns
        -------
        Mapping[str, CoverageData]
            Coverage data by file path.
        """
        json_path = coverage_file.with_suffix(".json")

        result = await self.tool_runner.run(
            "coverage",
            ["json", "-o", str(json_path), f"--data-file={coverage_file}"],
            coverage_file.parent,
        )

        if not result.success or not json_path.exists():
            log.warning("Failed to export coverage: %s", result.stderr)
            return {}

        try:
            data = json.loads(json_path.read_text())
            return RealCoverageCollector._parse_coverage_json(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to parse coverage JSON: %s", e)
            return {}

    @staticmethod
    def _parse_coverage_json(
        data: dict[str, object],
    ) -> Mapping[str, CoverageData]:
        """Parse coverage.py JSON output.

        Parameters
        ----------
        data
            Parsed coverage JSON.

        Returns
        -------
        Mapping[str, CoverageData]
            Coverage data by file path.
        """
        result: dict[str, CoverageData] = {}
        files = data.get("files", {})
        if not isinstance(files, dict):
            return result

        for path, file_data in files.items():
            if not isinstance(file_data, dict):
                continue
            executed = file_data.get("executed_lines", [])
            missing = file_data.get("missing_lines", [])
            excluded = file_data.get("excluded_lines", [])

            result[path] = CoverageData(
                path=path,
                covered_lines=frozenset(executed) if isinstance(executed, list) else frozenset(),
                missing_lines=frozenset(missing) if isinstance(missing, list) else frozenset(),
                excluded_lines=frozenset(excluded) if isinstance(excluded, list) else frozenset(),
            )

        return result


@dataclass
class RealTestReporter:
    """Production test reporter using pytest JSON reports."""

    async def collect(
        self,
        report_path: Path,
    ) -> tuple[TestResult, ...]:
        """Collect test results from pytest JSON report.

        Parameters
        ----------
        report_path
            Path to pytest JSON report.

        Returns
        -------
        tuple[TestResult, ...]
            Collected test results.
        """
        _ = self
        exists = await asyncio.to_thread(report_path.exists)
        if not exists:
            log.warning("Pytest report not found: %s", report_path)
            return ()

        try:
            text_content = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
            data = json.loads(text_content)
            return RealTestReporter._parse_pytest_json(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to parse pytest report: %s", e)
            return ()

    @staticmethod
    def _parse_pytest_json(
        data: dict[str, object],
    ) -> tuple[TestResult, ...]:
        """Parse pytest JSON report.

        Parameters
        ----------
        data
            Parsed pytest JSON.

        Returns
        -------
        tuple[TestResult, ...]
            Extracted test results.
        """
        results: list[TestResult] = []
        tests = data.get("tests", [])
        if not isinstance(tests, list):
            return ()

        for test in tests:
            if not isinstance(test, dict):
                continue
            node_id = str(test.get("nodeid", ""))

            if "::" in node_id:
                path, name = node_id.rsplit("::", 1)
            else:
                path = node_id
                name = ""

            results.append(
                TestResult(
                    node_id=node_id,
                    name=name,
                    path=path,
                    outcome=str(test.get("outcome", "unknown")),
                    duration_ms=int(float(test.get("duration", 0)) * 1000),
                    error_message=test.get("call", {}).get("longrepr")
                    if isinstance(test.get("call"), dict)
                    else None,
                    markers=tuple(str(m) for m in test.get("keywords", {}))
                    if isinstance(test.get("keywords"), dict)
                    else (),
                )
            )

        return tuple(results)


@dataclass
class RealGitHistoryProvider:
    """Production git history provider.

    Attributes
    ----------
    tool_runner
        Tool runner for subprocess execution.
    """

    tool_runner: SubprocessToolRunner

    async def log(
        self,
        repo_root: Path,
        *,
        path: Path | None = None,
        max_count: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[GitLogEntry, ...]:
        """Get git log entries.

        Parameters
        ----------
        repo_root
            Repository root directory.
        path
            Optional path to filter commits.
        max_count
            Maximum number of commits.
        since
            Only commits after this date.
        until
            Only commits before this date.

        Returns
        -------
        tuple[GitLogEntry, ...]
            Log entries.
        """
        format_str = "%H|%an|%ae|%aI|%s"

        args = [
            "log",
            f"--format={format_str}",
            "--shortstat",
        ]

        if max_count:
            args.append(f"-n{max_count}")
        if since:
            args.append(f"--since={since}")
        if until:
            args.append(f"--until={until}")
        if path:
            args.extend(["--", str(path)])

        result = await self.tool_runner.run("git", args, repo_root)

        if not result.success:
            log.warning("Git log failed: %s", result.stderr)
            return ()

        return RealGitHistoryProvider._parse_git_log(result.stdout)

    @staticmethod
    def _parse_git_log(output: str) -> tuple[GitLogEntry, ...]:
        """Parse git log output.

        Parameters
        ----------
        output
            Raw git log output.

        Returns
        -------
        tuple[GitLogEntry, ...]
            Parsed log entries.
        """
        entries: list[GitLogEntry] = []
        lines = output.strip().split("\n")
        idx = 0

        while idx < len(lines):
            line = lines[idx].strip()
            if not line or "|" not in line:
                idx += 1
                continue

            parts = line.split("|", 4)
            if len(parts) < _GIT_LOG_RECORD_PARTS:
                idx += 1
                continue

            sha, author, email, date, message = parts
            stat_info = _parse_git_stat_line(lines, idx)
            if stat_info[0]:
                idx += 1

            entries.append(
                GitLogEntry(
                    sha=sha,
                    author=author,
                    author_email=email,
                    date=date,
                    message=message,
                    files_changed=stat_info[1],
                    insertions=stat_info[2],
                    deletions=stat_info[3],
                )
            )
            idx += 1

        return tuple(entries)

    async def blame(
        self,
        repo_root: Path,
        path: Path,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> Mapping[int, GitLogEntry]:
        """Get git blame for a file.

        Parameters
        ----------
        repo_root
            Repository root directory.
        path
            File path.
        start_line
            Optional start line.
        end_line
            Optional end line.

        Returns
        -------
        Mapping[int, GitLogEntry]
            Line to commit mapping.
        """
        args = ["blame", "--porcelain"]

        if start_line and end_line:
            args.extend([f"-L{start_line},{end_line}"])

        args.append(str(path))

        result = await self.tool_runner.run("git", args, repo_root)

        if not result.success:
            log.warning("Git blame failed: %s", result.stderr)
            return {}

        return RealGitHistoryProvider._parse_git_blame(result.stdout)

    @staticmethod
    def _parse_git_blame(output: str) -> Mapping[int, GitLogEntry]:
        """Parse git blame porcelain output.

        Parameters
        ----------
        output
            Raw git blame output.

        Returns
        -------
        Mapping[int, GitLogEntry]
            Line to commit mapping.
        """
        return _parse_blame_output(output.strip().split("\n"))


@dataclass
class Providers:
    """Container for all DI providers.

    This class holds all the provider implementations and is passed
    to the BuildExecutor for wiring into TargetExecutionContext.

    Attributes
    ----------
    tool_runner
        Subprocess tool runner.
    scip_indexer
        SCIP index generator.
    type_checker
        Static type checker.
    coverage_collector
        Coverage data collector.
    test_reporter
        Test result collector.
    git_history
        Git history provider.
    """

    tool_runner: SubprocessToolRunner
    scip_indexer: RealScipIndexer
    type_checker: RealTypeChecker
    coverage_collector: RealCoverageCollector
    test_reporter: RealTestReporter
    git_history: RealGitHistoryProvider


def create_default_providers(tools_config: ToolsConfig) -> Providers:
    """Create a complete set of production providers.

    Parameters
    ----------
    tools_config
        Tool configuration with binary paths.

    Returns
    -------
    Providers
        Container with all providers wired together.
    """
    tool_runner = SubprocessToolRunner(tools_config)

    return Providers(
        tool_runner=tool_runner,
        scip_indexer=RealScipIndexer(tool_runner),
        type_checker=RealTypeChecker(tool_runner),
        coverage_collector=RealCoverageCollector(tool_runner),
        test_reporter=RealTestReporter(),
        git_history=RealGitHistoryProvider(tool_runner),
    )
