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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    # Types (re-exported from types.py for backward compatibility)
    "CoverageData",
    "GitLogEntry",
    "ScipIndexResult",
    "ScipOccurrence",
    "ScipParseResult",
    "ScipSymbol",
    "TestResult",
    "ToolRunResult",
    "TypeCheckResult",
    "TypeDiagnostic",
    # Protocols
    "CoverageCollector",
    "GitHistoryProvider",
    "ScipIndexer",
    "TestReporter",
    "ToolRunner",
    "TypeChecker",
]


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
