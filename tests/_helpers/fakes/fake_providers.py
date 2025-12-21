"""Fake implementations of build system protocols for testing.

These fakes return pre-configured data without executing real tools,
enabling fast and deterministic tests.

Example
-------
>>> fake_scip = FakeScipIndexer(
...     symbols=[ScipSymbol("my_func", "my_func", "function")],
... )
>>> result = await fake_scip.index(Path("/repo"), Path("/output/index.scip"))
>>> result.success
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.analytics_resources import AnalyticsResourceRegistryProvider
from codeintel.build.types import ScipIndexResult, ScipParseResult, TypeCheckResult
from codeintel.ingestion.engine.service import ToolService
from tests._helpers.fakes.tools import FakeToolRunner as IngestionFakeToolRunner
from tests._helpers.records import (
    CallRecorder,
    CollectCall,
    GitBlameCall,
    GitLogCall,
    ScipIndexCall,
    ScipParseCall,
    TypeCheckCall,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.types import (
        CoverageData,
        GitLogEntry,
        ScipOccurrence,
        ScipSymbol,
        TestResult,
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


@dataclass
class FakeToolRunner(IngestionFakeToolRunner):
    """Compatibility alias for the ingestion FakeToolRunner."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir=cache_dir or Path("build") / ".tool_cache")


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
    index_calls: CallRecorder[ScipIndexCall] = field(default_factory=CallRecorder)
    parse_calls: CallRecorder[ScipParseCall] = field(default_factory=CallRecorder)

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
        self.index_calls.record(
            ScipIndexCall(
                repo_root=repo_root,
                output_path=output_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
        )
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
        self.parse_calls.record(
            ScipParseCall(scip_path=scip_path, output_json_path=output_json_path)
        )
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
    calls: CallRecorder[TypeCheckCall] = field(default_factory=CallRecorder)

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
        self.calls.record(TypeCheckCall(repo_root=repo_root, paths=paths, config_path=config_path))
        error_count = sum(1 for d in self.diagnostics if d.severity == "error")
        warning_count = sum(1 for d in self.diagnostics if d.severity == "warning")

        return TypeCheckResult(
            success=self.success and error_count == 0,
            diagnostics=self.diagnostics,
            error_count=error_count,
            warning_count=warning_count,
            duration_ms=50,
        )


@dataclass
class FakeCoverageCollector:
    """Fake coverage collector that returns pre-configured data.

    Attributes
    ----------
    coverage_data
        Coverage data by file path.
    """

    coverage_data: dict[str, CoverageData] = field(default_factory=dict)
    collect_calls: CallRecorder[CollectCall] = field(default_factory=CallRecorder)

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
        self.collect_calls.record(CollectCall(path=coverage_file))
        return self.coverage_data


@dataclass
class FakeTestReporter:
    """Fake test reporter that returns pre-configured results.

    Attributes
    ----------
    test_results
        Test results to return.
    """

    test_results: tuple[TestResult, ...] = ()
    collect_calls: CallRecorder[CollectCall] = field(default_factory=CallRecorder)

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
        self.collect_calls.record(CollectCall(path=report_path))
        return self.test_results


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
    log_calls: CallRecorder[GitLogCall] = field(default_factory=CallRecorder)
    blame_calls: CallRecorder[GitBlameCall] = field(default_factory=CallRecorder)

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
        self.log_calls.record(
            GitLogCall(
                repo_root=repo_root,
                path=path,
                max_count=max_count,
                since=since,
                until=until,
            )
        )
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
        self.blame_calls.record(
            GitBlameCall(
                repo_root=repo_root,
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
        )
        return self.blame_data.get(str(path), {})


@dataclass
class FakeProviders:
    """Container for fake tool providers aligned with BuildEnv."""

    tool_runner: FakeToolRunner = field(default_factory=FakeToolRunner)
    resources: AnalyticsResourceRegistryProvider = field(
        default_factory=AnalyticsResourceRegistryProvider
    )
    tool_service: ToolService = field(init=False)

    def __post_init__(self) -> None:
        self.tool_service = ToolService(self.tool_runner)

    @classmethod
    def defaults(cls) -> FakeProviders:
        """Create providers with all defaults.

        Returns
        -------
        FakeProviders
            Container with default fake instances.
        """
        return cls()
