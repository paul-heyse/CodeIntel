"""Tool port protocol for external analysis tool execution.

This module defines the port protocol for executing external analysis tools
like pyright, ruff, and scip-python. The protocol abstracts
tool invocation details to enable testing without real tool installations.

Architecture Note
-----------------
This module defines **port interface types** (DiagnosticResult, ScipResult,
TestResult) that represent the contract between ingestion steps
and tool adapters. These are intentionally simpler than the richer "Report"
types in ``tools/results.py`` which are used internally by tool plugins.

The ``ToolRunnerAdapter`` converts from Report types to these Result types
at the port boundary, providing a clean interface while preserving rich
internal representations for tool plugin logic.

See Also
--------
codeintel.ingestion.engine.results : Rich domain types for tool plugin internals
codeintel.ingestion.adapters.tool_runner : Adapter that bridges the layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.ingestion.engine.status import ToolStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class DiagnosticEntry:
    """A single diagnostic entry from a type checker or linter.

    Attributes
    ----------
    path
        Relative file path.
    line
        Line number (1-based).
    column
        Column number (1-based).
    severity
        Diagnostic severity (error, warning, info).
    code
        Diagnostic code or rule identifier.
    message
        Human-readable message.
    """

    path: str
    line: int
    column: int
    severity: str
    code: str
    message: str


@dataclass
class DiagnosticResult:
    """Result from running a diagnostic tool (pyright, ruff, pyrefly).

    Attributes
    ----------
    status
        Execution status.
    diagnostics
        List of diagnostic entries.
    error
        Error message if status is FAILED.
    duration_s
        Execution duration in seconds.
    """

    status: ToolStatus
    diagnostics: list[DiagnosticEntry] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    def errors_by_path(self) -> dict[str, int]:
        """Return error counts grouped by file path.

        Returns
        -------
        dict[str, int]
            Mapping from relative path to error count.
        """
        counts: dict[str, int] = {}
        for diag in self.diagnostics:
            if diag.severity == "error":
                counts[diag.path] = counts.get(diag.path, 0) + 1
        return counts


@dataclass(frozen=True)
class ScipSymbol:
    """A symbol from SCIP index.

    Attributes
    ----------
    symbol
        SCIP symbol identifier.
    documentation
        Optional documentation string.
    """

    symbol: str
    documentation: str | None = None


@dataclass(frozen=True)
class ScipOccurrence:
    """A symbol occurrence from SCIP index.

    Attributes
    ----------
    symbol
        SCIP symbol identifier.
    range_start_line
        Start line (0-based).
    range_start_col
        Start column (0-based).
    range_end_line
        End line (0-based).
    range_end_col
        End column (0-based).
    symbol_roles
        Bitmask of symbol roles (1=definition, 2=reference).
    syntax_kind
        Optional syntax highlighting kind enum.
    enclosing_start_line
        Enclosing range start line (0-based).
    enclosing_start_col
        Enclosing range start column (0-based).
    enclosing_end_line
        Enclosing range end line (0-based).
    enclosing_end_col
        Enclosing range end column (0-based).
    override_documentation
        Optional occurrence-specific documentation.
    position_encoding
        Encoding enum for interpreting column offsets.
    text_document_encoding
        Text encoding for source files on disk.
    start_byte
        Start byte offset, when computed.
    end_byte
        End byte offset, when computed.
    """

    symbol: str
    range_start_line: int
    range_start_col: int
    range_end_line: int
    range_end_col: int
    symbol_roles: int
    syntax_kind: int | None = None
    enclosing_start_line: int | None = None
    enclosing_start_col: int | None = None
    enclosing_end_line: int | None = None
    enclosing_end_col: int | None = None
    override_documentation: str | None = None
    position_encoding: int | None = None
    text_document_encoding: str | None = None
    start_byte: int | None = None
    end_byte: int | None = None


@dataclass
class ScipDocument:
    """A document from SCIP index.

    Attributes
    ----------
    relative_path
        Relative file path.
    symbols
        Symbols defined in this document.
    occurrences
        Symbol occurrences in this document.
    position_encoding
        Encoding enum for interpreting column offsets.
    text_document_encoding
        Text encoding for source files on disk.
    """

    relative_path: str
    symbols: Sequence[ScipSymbol] = field(default_factory=list)
    occurrences: Sequence[ScipOccurrence] = field(default_factory=list)
    position_encoding: int | None = None
    text_document_encoding: str | None = None


@dataclass
class ScipResult:
    """Result from running SCIP indexing.

    Attributes
    ----------
    status
        Execution status.
    documents
        SCIP documents indexed.
    index_scip_path
        Path to the SCIP index file.
    error
        Error message if status is FAILED.
    duration_s
        Execution duration in seconds.
    """

    status: ToolStatus
    documents: list[ScipDocument] = field(default_factory=list)
    index_scip_path: Path | None = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class ScipRunRequest:
    """Request payload for running SCIP indexing.

    Attributes
    ----------
    repo_root
        Repository root directory.
    output_scip
        Path for SCIP index output.
    proto_module_path
        Path to generated scip_pb2 module.
    target_dir
        Optional repo subdirectory to index.
    rel_paths
        Optional repo-relative paths to index.
    environment_json
        Optional environment JSON passed to scip-python.
    project_version
        Optional project version passed to scip-python.
    project_namespace
        Optional namespace prefix passed to scip-python.
    timeout_s
        Optional timeout override (seconds).
    """

    repo_root: Path
    output_scip: Path
    proto_module_path: Path
    target_dir: Path | None = None
    rel_paths: Sequence[str] | None = None
    environment_json: Path | None = None
    project_version: str | None = None
    project_namespace: str | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class TestCase:
    """A single test case result.

    Attributes
    ----------
    nodeid
        Pytest node identifier.
    outcome
        Test outcome (passed, failed, skipped, error).
    duration_s
        Test duration in seconds.
    longrepr
        Long representation of failure if applicable.
    """

    nodeid: str
    outcome: str
    duration_s: float = 0.0
    longrepr: str | None = None


@dataclass
class TestResult:
    """Result from running pytest.

    Attributes
    ----------
    status
        Execution status.
    tests
        Individual test case results.
    passed
        Number of passed tests.
    failed
        Number of failed tests.
    skipped
        Number of skipped tests.
    error
        Error message if status is FAILED.
    duration_s
        Total execution duration in seconds.
    """

    status: ToolStatus
    tests: list[TestCase] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: str | None = None
    duration_s: float = 0.0


@runtime_checkable
class IngestToolPort(Protocol):
    """Port protocol for executing external analysis tools.

    This protocol abstracts tool execution to enable testing without real
    tool installations and to normalize result formats across tools.
    """

    async def run_pyright(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run pyright type checker.

        Parameters
        ----------
        repo_root
            Repository root directory.
        paths
            Optional paths to scope diagnostics (relative to repo_root or absolute).

        Returns
        -------
        DiagnosticResult
            Type checking results with diagnostics.
        """
        ...

    async def run_pyrefly(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run pyrefly type checker.

        Parameters
        ----------
        repo_root
            Repository root directory.
        paths
            Optional paths to scope diagnostics (relative to repo_root or absolute).

        Returns
        -------
        DiagnosticResult
            Type checking results with diagnostics.
        """
        ...

    async def run_ruff(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run ruff linter.

        Parameters
        ----------
        repo_root
            Repository root directory.
        paths
            Optional paths to scope diagnostics (relative to repo_root or absolute).

        Returns
        -------
        DiagnosticResult
            Linting results with diagnostics.
        """
        ...

    async def run_scip(self, request: ScipRunRequest) -> ScipResult:
        """Run SCIP indexing.

        Parameters
        ----------
        request
            SCIP run request payload.

        Returns
        -------
        ScipResult
            SCIP indexing results.
        """
        ...


__all__ = [
    "DiagnosticEntry",
    "DiagnosticResult",
    "IngestToolPort",
    "ScipDocument",
    "ScipOccurrence",
    "ScipResult",
    "ScipRunRequest",
    "ScipSymbol",
    "TestCase",
    "TestResult",
    "ToolStatus",
]
