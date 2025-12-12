"""Protocol definitions for dependency injection in the build system.

This module defines the interfaces (protocols) for external dependencies
that plugins need. By programming to protocols rather than concrete
implementations, we achieve:

1. **Testability**: Tests inject fake implementations
2. **Flexibility**: Production can swap implementations
3. **Clarity**: Protocols document what's needed

Plugins receive implementations via TargetExecutionContext.resources,
and don't know or care whether they're real or fake.

Example
-------
>>> from codeintel.build.protocols import ToolRunner, ToolRunResult
>>> class FakeToolRunner:
...     async def run(self, tool: str, args: list[str], cwd: Path) -> ToolRunResult:
...         return ToolRunResult(tool=tool, returncode=0, stdout="", stderr="")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "CoverageCollector",
    "CoverageData",
    "GitHistoryProvider",
    "GitLogEntry",
    "ScipIndexResult",
    "ScipIndexer",
    "ScipOccurrence",
    "ScipParseResult",
    "ScipSymbol",
    "TestReporter",
    "TestResult",
    "ToolRunResult",
    "ToolRunner",
    "TypeCheckResult",
    "TypeChecker",
    "TypeDiagnostic",
]


@dataclass(frozen=True)
class ToolRunResult:
    """Result of running an external tool.

    Attributes
    ----------
    tool
        Name of the tool that was run.
    args
        Arguments passed to the tool.
    returncode
        Exit code (0 = success).
    stdout
        Standard output content.
    stderr
        Standard error content.
    duration_ms
        Execution time in milliseconds.
    """

    tool: str
    args: tuple[str, ...] = ()
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0

    @property
    def success(self) -> bool:
        """Check if the tool succeeded.

        Returns
        -------
        bool
            True if returncode is 0.
        """
        return self.returncode == 0


@runtime_checkable
class ToolRunner(Protocol):
    """Interface for running external tools.

    This protocol abstracts subprocess execution, allowing tests
    to inject fake runners that return canned results.

    Production implementations use asyncio.create_subprocess_exec
    to run actual binaries.
    """

    async def run(
        self,
        tool: str,
        args: Sequence[str],
        cwd: Path,
        *,
        timeout_ms: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ToolRunResult:
        """Run an external tool.

        Parameters
        ----------
        tool
            Tool name (e.g., "scip-python", "pyright").
        args
            Command-line arguments.
        cwd
            Working directory.
        timeout_ms
            Optional timeout in milliseconds.
        env
            Optional environment variables to add.

        Returns
        -------
        ToolRunResult
            Result with exit code, stdout, stderr.
        """
        ...

    def is_available(self, tool: str) -> bool:
        """Check if a tool is available on the system.

        Parameters
        ----------
        tool
            Tool name to check.

        Returns
        -------
        bool
            True if the tool can be executed.
        """
        ...


@dataclass(frozen=True)
class ScipSymbol:
    """A symbol extracted from SCIP index.

    Attributes
    ----------
    symbol
        SCIP symbol string (unique identifier).
    name
        Simple name of the symbol.
    kind
        Symbol kind (function, class, variable, etc.).
    documentation
        Optional documentation string.
    signature
        Optional type signature.
    """

    symbol: str
    name: str
    kind: str
    documentation: str | None = None
    signature: str | None = None


@dataclass(frozen=True)
class ScipOccurrence:
    """A symbol occurrence in source code.

    Attributes
    ----------
    symbol
        SCIP symbol string this occurrence refers to.
    path
        File path relative to repo root.
    line
        Line number (1-based).
    character
        Character offset (0-based).
    end_line
        End line number.
    end_character
        End character offset.
    role
        Occurrence role (definition, reference, etc.).
    """

    symbol: str
    path: str
    line: int
    character: int
    end_line: int
    end_character: int
    role: str


@dataclass(frozen=True)
class ScipIndexResult:
    """Result of SCIP index generation.

    Attributes
    ----------
    success
        Whether indexing succeeded.
    index_path
        Path to generated index.scip file.
    error_message
        Error message if failed.
    duration_ms
        Execution time.
    """

    success: bool
    index_path: Path | None = None
    error_message: str | None = None
    duration_ms: int = 0


@dataclass(frozen=True)
class ScipParseResult:
    """Result of parsing SCIP index to JSON.

    Attributes
    ----------
    success
        Whether parsing succeeded.
    symbols
        Extracted symbols.
    occurrences
        Symbol occurrences.
    json_path
        Path to generated JSON file.
    error_message
        Error message if failed.
    """

    success: bool
    symbols: tuple[ScipSymbol, ...] = ()
    occurrences: tuple[ScipOccurrence, ...] = ()
    json_path: Path | None = None
    error_message: str | None = None


@runtime_checkable
class ScipIndexer(Protocol):
    """Interface for SCIP index generation and parsing.

    This protocol abstracts the scip-python and scip tools,
    allowing tests to inject fake indexers with canned data.
    """

    async def index(
        self,
        repo_root: Path,
        output_path: Path,
        *,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> ScipIndexResult:
        """Generate SCIP index for a repository.

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
            Result with success status and index path.
        """
        ...

    async def parse(
        self,
        scip_path: Path,
        output_json_path: Path,
    ) -> ScipParseResult:
        """Parse SCIP index to JSON format.

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
        ...


@dataclass(frozen=True)
class TypeDiagnostic:
    """A diagnostic from static type checking.

    Attributes
    ----------
    path
        File path relative to repo root.
    line
        Line number (1-based).
    character
        Character offset (0-based).
    severity
        Diagnostic severity (error, warning, info).
    code
        Diagnostic code (e.g., "reportGeneralTypeIssues").
    message
        Human-readable diagnostic message.
    source
        Source tool (pyright, pyrefly, ruff).
    """

    path: str
    line: int
    character: int
    severity: str
    code: str
    message: str
    source: str


@dataclass(frozen=True)
class TypeCheckResult:
    """Result of type checking a codebase.

    Attributes
    ----------
    success
        Whether type checking passed (no errors).
    diagnostics
        List of diagnostics found.
    error_count
        Number of error-level diagnostics.
    warning_count
        Number of warning-level diagnostics.
    duration_ms
        Execution time.
    """

    success: bool
    diagnostics: tuple[TypeDiagnostic, ...] = ()
    error_count: int = 0
    warning_count: int = 0
    duration_ms: int = 0


@runtime_checkable
class TypeChecker(Protocol):
    """Interface for static type checking.

    This protocol abstracts pyright, pyrefly, and ruff,
    allowing tests to inject fake checkers with canned diagnostics.
    """

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> TypeCheckResult:
        """Run type checking on the codebase.

        Parameters
        ----------
        repo_root
            Repository root directory.
        paths
            Optional specific paths to check.
        config_path
            Optional path to tool config file.

        Returns
        -------
        TypeCheckResult
            Result with diagnostics and counts.
        """
        ...


@dataclass(frozen=True)
class CoverageData:
    """Coverage data for a file.

    Attributes
    ----------
    path
        File path relative to repo root.
    covered_lines
        Set of line numbers that are covered.
    missing_lines
        Set of line numbers that are not covered.
    excluded_lines
        Set of lines excluded from coverage.
    branch_coverage
        Optional branch coverage percentage.
    """

    path: str
    covered_lines: frozenset[int] = field(default_factory=frozenset)
    missing_lines: frozenset[int] = field(default_factory=frozenset)
    excluded_lines: frozenset[int] = field(default_factory=frozenset)
    branch_coverage: float | None = None

    @property
    def line_coverage(self) -> float:
        """Calculate line coverage percentage.

        Returns
        -------
        float
            Coverage percentage (0-100).
        """
        total = len(self.covered_lines) + len(self.missing_lines)
        if total == 0:
            return 100.0
        return (len(self.covered_lines) / total) * 100


@runtime_checkable
class CoverageCollector(Protocol):
    """Interface for collecting test coverage data.

    This protocol abstracts coverage.py and pytest-cov,
    allowing tests to inject fake coverage data.
    """

    async def collect(
        self,
        coverage_file: Path,
    ) -> Mapping[str, CoverageData]:
        """Collect coverage data from a coverage file.

        Parameters
        ----------
        coverage_file
            Path to coverage data file (.coverage or coverage.json).

        Returns
        -------
        Mapping[str, CoverageData]
            Mapping of file path to coverage data.
        """
        ...


@dataclass(frozen=True)
class TestResult:
    """Result of a single test.

    Attributes
    ----------
    node_id
        Pytest node ID (e.g., "tests/test_foo.py::test_bar").
    name
        Test function name.
    path
        Test file path.
    outcome
        Test outcome (passed, failed, skipped, error).
    duration_ms
        Test execution time.
    error_message
        Error message if failed.
    markers
        Pytest markers applied to test.
    """

    node_id: str
    name: str
    path: str
    outcome: str
    duration_ms: int = 0
    error_message: str | None = None
    markers: tuple[str, ...] = ()


@runtime_checkable
class TestReporter(Protocol):
    """Interface for collecting test results.

    This protocol abstracts pytest report parsing,
    allowing tests to inject fake test results.
    """

    async def collect(
        self,
        report_path: Path,
    ) -> tuple[TestResult, ...]:
        """Collect test results from a pytest report.

        Parameters
        ----------
        report_path
            Path to pytest JSON report.

        Returns
        -------
        tuple[TestResult, ...]
            Collected test results.
        """
        ...


@dataclass(frozen=True)
class GitLogEntry:
    """A git commit log entry.

    Attributes
    ----------
    sha
        Commit SHA.
    author
        Author name.
    author_email
        Author email.
    date
        Commit date as ISO string.
    message
        Commit message (first line).
    files_changed
        Number of files changed.
    insertions
        Lines inserted.
    deletions
        Lines deleted.
    """

    sha: str
    author: str
    author_email: str
    date: str
    message: str
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


@runtime_checkable
class GitHistoryProvider(Protocol):
    """Interface for git history queries.

    This protocol abstracts git log and blame commands,
    allowing tests to inject fake history data.
    """

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
            Maximum number of commits to return.
        since
            Only commits after this date.
        until
            Only commits before this date.

        Returns
        -------
        tuple[GitLogEntry, ...]
            Log entries in reverse chronological order.
        """
        ...

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
            File path relative to repo root.
        start_line
            Optional start line (1-based).
        end_line
            Optional end line.

        Returns
        -------
        Mapping[int, GitLogEntry]
            Mapping of line number to commit that last changed it.
        """
        ...
