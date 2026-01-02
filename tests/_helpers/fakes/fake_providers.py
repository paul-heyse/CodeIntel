"""Fake implementations of build system protocols for testing.

These fakes return pre-configured data without executing real tools,
enabling fast and deterministic tests.

Example
-------
>>> fake_scip = FakeScipIndexer(
...     symbols=[ScipSymbol("my_func")],
... )
>>> result = await fake_scip.index(Path("/repo"), Path("/output/index.scip"))
>>> result.success
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.engine.status import ToolStatus
from codeintel.ingestion.ports.tools import (
    DiagnosticEntry,
    DiagnosticResult,
    ScipDocument,
    ScipOccurrence,
    ScipResult,
    ScipSymbol,
    TestCase,
)
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

__all__ = [
    "FakeGitHistoryProvider",
    "FakeProviders",
    "FakeScipIndexer",
    "FakeTestReporter",
    "FakeToolRunner",
    "FakeTypeChecker",
]


@dataclass(frozen=True)
class GitLogEntry:
    """Minimal git log entry used in tests."""

    sha: str
    author: str
    author_email: str
    date: str
    message: str
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


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
    ) -> ScipResult:
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
        ScipResult
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
            return ScipResult(
                status=ToolStatus.OK,
                documents=[],
                index_scip_path=output_path,
                duration_s=0.1,
            )
        return ScipResult(
            status=ToolStatus.FAILED,
            documents=[],
            error="Fake SCIP indexing failed",
            duration_s=0.0,
        )

    async def parse(
        self,
        scip_path: Path,
        output_json_path: Path,
    ) -> ScipResult:
        """Return pre-configured parse result.

        Parameters
        ----------
        scip_path
            SCIP path.
        output_json_path
            Output path to report.

        Returns
        -------
        ScipResult
            Pre-configured result with symbols and occurrences.
        """
        self.parse_calls.record(
            ScipParseCall(scip_path=scip_path, output_json_path=output_json_path)
        )
        if self.parse_success:
            document = ScipDocument(
                relative_path=output_json_path.name,
                symbols=list(self.symbols),
                occurrences=list(self.occurrences),
            )
            return ScipResult(
                status=ToolStatus.OK,
                documents=[document],
                index_scip_path=scip_path,
                duration_s=0.1,
            )
        return ScipResult(
            status=ToolStatus.FAILED,
            documents=[],
            index_scip_path=scip_path,
            error="Fake SCIP parsing failed",
            duration_s=0.0,
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

    diagnostics: tuple[DiagnosticEntry, ...] = ()
    success: bool = True
    calls: CallRecorder[TypeCheckCall] = field(default_factory=CallRecorder)

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> DiagnosticResult:
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
        DiagnosticResult
            Pre-configured result.
        """
        self.calls.record(TypeCheckCall(repo_root=repo_root, paths=paths, config_path=config_path))
        error_count = sum(1 for d in self.diagnostics if d.severity == "error")
        status = ToolStatus.OK if self.success and error_count == 0 else ToolStatus.FAILED
        error_message = None if status is ToolStatus.OK else "Fake type checking failed"

        return DiagnosticResult(
            status=status,
            diagnostics=list(self.diagnostics),
            error=error_message,
            duration_s=0.05,
        )


@dataclass
class FakeTestReporter:
    """Fake test reporter that returns pre-configured results.

    Attributes
    ----------
    test_results
        Test results to return.
    """

    test_results: tuple[TestCase, ...] = ()
    collect_calls: CallRecorder[CollectCall] = field(default_factory=CallRecorder)

    async def collect(
        self,
        report_path: Path,
    ) -> tuple[TestCase, ...]:
        """Return pre-configured test results.

        Parameters
        ----------
        report_path
            Report path.

        Returns
        -------
        tuple[TestCase, ...]
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
