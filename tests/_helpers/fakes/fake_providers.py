"""Fake implementations of build system protocols for testing.

These fakes return pre-configured data without executing real tools,
enabling fast and deterministic tests.

Example
-------
>>> fake_scip = FakeScipIndexer(
...     symbols=[ScipSymbol("my_func", "my_func", "function")],
... )
>>> result = await fake_scip.index(Path("/repo"), Path("/output/index.scip"))
>>> assert result.success
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.build.protocols import (
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

__all__ = [
    "FakeCoverageCollector",
    "FakeGitHistoryProvider",
    "FakeProviders",
    "FakeScipIndexer",
    "FakeTestReporter",
    "FakeToolRunner",
    "FakeTypeChecker",
]


# =============================================================================
# Fake Tool Runner
# =============================================================================


@dataclass
class FakeToolRunner:
    """Fake tool runner that returns pre-configured results.

    Attributes
    ----------
    available_tools
        Set of tools considered "available".
    results
        Pre-configured results by tool name.
    default_result
        Default result for tools not in results dict.
    call_log
        Record of all calls made (for verification).
    """

    available_tools: set[str] = field(default_factory=lambda: {"git", "python"})
    results: dict[str, ToolRunResult] = field(default_factory=dict)
    default_result: ToolRunResult = field(
        default_factory=lambda: ToolRunResult(tool="unknown", returncode=0)
    )
    call_log: list[tuple[str, Sequence[str], Path, int | None, Mapping[str, str] | None]] = field(
        default_factory=list
    )

    async def run(
        self,
        tool: str,
        args: Sequence[str],
        cwd: Path,
        *,
        timeout_ms: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ToolRunResult:
        """Return pre-configured result for tool.

        Parameters
        ----------
        tool
            Tool name.
        args
            Command-line arguments.
        cwd
            Working directory.
        timeout_ms
            Timeout for tool execution.
        env
            Environment variables to pass.

        Returns
        -------
        ToolRunResult
            Pre-configured or default result.
        """
        env_snapshot = dict(env) if env is not None else None
        self.call_log.append((tool, list(args), cwd, timeout_ms, env_snapshot))
        return self.results.get(tool, self.default_result)

    def is_available(self, tool: str) -> bool:
        """Check if tool is in available set.

        Parameters
        ----------
        tool
            Tool name.

        Returns
        -------
        bool
            True if in available_tools.
        """
        return tool in self.available_tools


# =============================================================================
# Fake SCIP Indexer
# =============================================================================


@dataclass
class FakeScipIndexer:
    """Fake SCIP indexer that returns pre-configured symbols.

    Attributes
    ----------
    symbols
        Symbols to return from parse().
    occurrences
        Occurrences to return from parse().
    index_success
        Whether index() should succeed.
    parse_success
        Whether parse() should succeed.
    """

    symbols: tuple[ScipSymbol, ...] = ()
    occurrences: tuple[ScipOccurrence, ...] = ()
    index_success: bool = True
    parse_success: bool = True
    index_calls: list[
        tuple[Path, Path, Sequence[str] | None, Sequence[str] | None]
    ] = field(default_factory=list)
    parse_calls: list[tuple[Path, Path]] = field(default_factory=list)

    async def index(
        self,
        repo_root: Path,
        output_path: Path,
        *,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> ScipIndexResult:
        """Return pre-configured index result.

        Parameters
        ----------
        repo_root
            Repository root.
        output_path
            Output path to report.
        include_patterns
            Include patterns.
        exclude_patterns
            Exclude patterns.

        Returns
        -------
        ScipIndexResult
            Pre-configured result.
        """
        self.index_calls.append((repo_root, output_path, include_patterns, exclude_patterns))
        if self.index_success:
            return ScipIndexResult(
                success=True,
                index_path=output_path,
                duration_ms=100,
            )
        return ScipIndexResult(
            success=False,
            error_message="Fake SCIP indexing failed",
        )

    async def parse(
        self,
        scip_path: Path,
        output_json_path: Path,
    ) -> ScipParseResult:
        """Return pre-configured parse result.

        Parameters
        ----------
        scip_path
            SCIP path.
        output_json_path
            Output path to report.

        Returns
        -------
        ScipParseResult
            Pre-configured result with symbols and occurrences.
        """
        self.parse_calls.append((scip_path, output_json_path))
        if self.parse_success:
            return ScipParseResult(
                success=True,
                symbols=self.symbols,
                occurrences=self.occurrences,
                json_path=output_json_path,
            )
        return ScipParseResult(
            success=False,
            error_message="Fake SCIP parsing failed",
        )


# =============================================================================
# Fake Type Checker
# =============================================================================


@dataclass
class FakeTypeChecker:
    """Fake type checker that returns pre-configured diagnostics.

    Attributes
    ----------
    diagnostics
        Diagnostics to return.
    success
        Whether check should succeed (no errors).
    """

    diagnostics: tuple[TypeDiagnostic, ...] = ()
    success: bool = True
    call_log: list[tuple[Path, Sequence[Path] | None, Path | None]] = field(
        default_factory=list
    )

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> TypeCheckResult:
        """Return pre-configured type check result.

        Parameters
        ----------
        repo_root
            Repository root.
        paths
            Paths to check.
        config_path
            Config path.

        Returns
        -------
        TypeCheckResult
            Pre-configured result.
        """
        self.call_log.append((repo_root, paths, config_path))
        error_count = sum(1 for d in self.diagnostics if d.severity == "error")
        warning_count = sum(1 for d in self.diagnostics if d.severity == "warning")

        return TypeCheckResult(
            success=self.success and error_count == 0,
            diagnostics=self.diagnostics,
            error_count=error_count,
            warning_count=warning_count,
            duration_ms=50,
        )


# =============================================================================
# Fake Coverage Collector
# =============================================================================


@dataclass
class FakeCoverageCollector:
    """Fake coverage collector that returns pre-configured data.

    Attributes
    ----------
    coverage_data
        Coverage data by file path.
    """

    coverage_data: dict[str, CoverageData] = field(default_factory=dict)
    collect_calls: list[Path] = field(default_factory=list)

    async def collect(
        self,
        coverage_file: Path,
    ) -> Mapping[str, CoverageData]:
        """Return pre-configured coverage data.

        Parameters
        ----------
        coverage_file
            Coverage file path.

        Returns
        -------
        Mapping[str, CoverageData]
            Pre-configured coverage data.
        """
        self.collect_calls.append(coverage_file)
        return self.coverage_data


# =============================================================================
# Fake Test Reporter
# =============================================================================


@dataclass
class FakeTestReporter:
    """Fake test reporter that returns pre-configured results.

    Attributes
    ----------
    test_results
        Test results to return.
    """

    test_results: tuple[TestResult, ...] = ()
    collect_calls: list[Path] = field(default_factory=list)

    async def collect(
        self,
        report_path: Path,
    ) -> tuple[TestResult, ...]:
        """Return pre-configured test results.

        Parameters
        ----------
        report_path
            Report path.

        Returns
        -------
        tuple[TestResult, ...]
            Pre-configured results.
        """
        self.collect_calls.append(report_path)
        return self.test_results


# =============================================================================
# Fake Git History Provider
# =============================================================================


@dataclass
class FakeGitHistoryProvider:
    """Fake git history provider that returns pre-configured data.

    Attributes
    ----------
    log_entries
        Log entries to return.
    blame_data
        Blame data by file path.
    """

    log_entries: tuple[GitLogEntry, ...] = ()
    blame_data: dict[str, dict[int, GitLogEntry]] = field(default_factory=dict)
    log_calls: list[tuple[Path, Path | None, int | None, str | None, str | None]] = field(
        default_factory=list
    )
    blame_calls: list[tuple[Path, Path, int | None, int | None]] = field(
        default_factory=list
    )

    async def log(
        self,
        repo_root: Path,
        *,
        path: Path | None = None,
        max_count: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[GitLogEntry, ...]:
        """Return pre-configured log entries.

        Parameters
        ----------
        repo_root
            Repository root.
        path
            Path filter.
        max_count
            Maximum entries.
        since
            Start date.
        until
            End date.

        Returns
        -------
        tuple[GitLogEntry, ...]
            Pre-configured entries (optionally limited).
        """
        self.log_calls.append((repo_root, path, max_count, since, until))
        entries = self.log_entries
        if max_count:
            entries = entries[:max_count]
        return entries

    async def blame(
        self,
        repo_root: Path,
        path: Path,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> Mapping[int, GitLogEntry]:
        """Return pre-configured blame data.

        Parameters
        ----------
        repo_root
            Repository root.
        path
            File path to look up.
        start_line
            Start line.
        end_line
            End line.

        Returns
        -------
        Mapping[int, GitLogEntry]
            Pre-configured blame data for path.
        """
        self.blame_calls.append((repo_root, path, start_line, end_line))
        return self.blame_data.get(str(path), {})


# =============================================================================
# Fake Providers Container
# =============================================================================


@dataclass
class FakeProviders:
    """Container for all fake providers.

    Use this in tests to provide a complete set of fakes.

    Attributes
    ----------
    tool_runner
        Fake tool runner.
    scip_indexer
        Fake SCIP indexer.
    type_checker
        Fake type checker.
    coverage_collector
        Fake coverage collector.
    test_reporter
        Fake test reporter.
    git_history
        Fake git history provider.
    """

    tool_runner: FakeToolRunner = field(default_factory=FakeToolRunner)
    scip_indexer: FakeScipIndexer = field(default_factory=FakeScipIndexer)
    type_checker: FakeTypeChecker = field(default_factory=FakeTypeChecker)
    coverage_collector: FakeCoverageCollector = field(default_factory=FakeCoverageCollector)
    test_reporter: FakeTestReporter = field(default_factory=FakeTestReporter)
    git_history: FakeGitHistoryProvider = field(default_factory=FakeGitHistoryProvider)

    @classmethod
    def defaults(cls) -> FakeProviders:
        """Create providers with all defaults.

        Returns
        -------
        FakeProviders
            Container with default fake instances.
        """
        return cls()
