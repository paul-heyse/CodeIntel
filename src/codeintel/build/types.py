"""Shared type definitions for the build system.

This module contains dataclasses representing results from external tools
and build operations. These types are used by both protocols (interfaces)
and providers (implementations).

Separating types from protocols enables:

1. **Cleaner imports**: Consumers can import just the types they need
2. **Reduced coupling**: Types don't depend on protocol definitions
3. **Better testability**: Tests can import types without protocol overhead

Example
-------
>>> from codeintel.build.types import ToolRunResult, TypeDiagnostic
>>> result = ToolRunResult(tool="pyright", returncode=0)
>>> result.success
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
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
