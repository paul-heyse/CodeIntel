"""Domain result types for tool plugin outputs.

This module defines structured result types that tool plugins return after
parsing raw tool output. These domain objects provide type safety and a
clear contract between tool plugins and ingestion consumers.

Architecture Note
-----------------
These "Report" types (DiagnosticReport, CoverageReport, TestReport, ScipIndexResult)
are **rich domain objects** used internally by tool plugins. They include:
- Aggregated counts (total_errors, total_warnings, definition_count, etc.)
- Factory methods for construction from raw data (from_error_counts, from_documents)
- Helper methods for common access patterns (errors_by_path, by_path, definitions_by_location)

In contrast, the "Result" types in ``ports/tools.py`` (DiagnosticResult, CoverageResult,
TestResult, ScipResult) are **simpler port interface types** with status/error/duration
fields suitable for clean architectural boundaries.

The ``ToolRunnerAdapter`` converts from these rich Report types to the simpler Result
types at the port boundary.

See Also
--------
codeintel.ingestion.ports.tools : Simpler port interface types
codeintel.ingestion.adapters.tool_runner : Adapter that bridges the layers
codeintel.ingestion.tool_service : Facade using these Report types internally
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.ingestion.ports import tools as port_types

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class ReportProtocol(Protocol):
    """Protocol for tool result report types with factory methods.

    All Report types (DiagnosticReport, CoverageReport, TestReport, ScipIndexResult)
    implement this protocol, providing a common interface for creating empty instances.
    """

    @staticmethod
    def empty() -> ReportProtocol:
        """Create an empty report instance.

        Returns
        -------
        ReportProtocol
            Empty report with no data.
        """
        ...


@dataclass(frozen=True)
class FileDiagnosticCount:
    """Diagnostic counts for a single file.

    Attributes
    ----------
    rel_path
        Repository-relative file path.
    error_count
        Number of errors in this file.
    warning_count
        Number of warnings in this file.
    """

    rel_path: str
    error_count: int = 0
    warning_count: int = 0


@dataclass(frozen=True)
class DiagnosticReport:
    """Aggregated diagnostic report from a static analysis tool.

    Used by pyright, pyrefly, and ruff plugins to return normalized
    diagnostic counts per file.

    Attributes
    ----------
    tool_name
        Name of the tool that generated this report.
    files
        Mapping of relative file paths to diagnostic counts.
    total_errors
        Sum of all error counts.
    total_warnings
        Sum of all warning counts.
    raw_output
        Optional raw JSON output for debugging.
    """

    tool_name: str
    files: Mapping[str, FileDiagnosticCount]
    total_errors: int = 0
    total_warnings: int = 0
    raw_output: str | None = None

    @classmethod
    def from_error_counts(
        cls,
        tool_name: str,
        errors_by_file: Mapping[str, int],
        *,
        warnings_by_file: Mapping[str, int] | None = None,
        raw_output: str | None = None,
    ) -> DiagnosticReport:
        """
        Build a DiagnosticReport from simple error/warning count mappings.

        Parameters
        ----------
        tool_name
            Name of the source tool.
        errors_by_file
            Mapping of relative path to error count.
        warnings_by_file
            Optional mapping of relative path to warning count.
        raw_output
            Optional raw output for debugging.

        Returns
        -------
        DiagnosticReport
            Constructed report with aggregated totals.
        """
        warnings = warnings_by_file or {}
        all_paths = set(errors_by_file) | set(warnings)
        files: dict[str, FileDiagnosticCount] = {}
        total_errors = 0
        total_warnings = 0

        for rel_path in all_paths:
            err_count = errors_by_file.get(rel_path, 0)
            warn_count = warnings.get(rel_path, 0)
            files[rel_path] = FileDiagnosticCount(
                rel_path=rel_path,
                error_count=err_count,
                warning_count=warn_count,
            )
            total_errors += err_count
            total_warnings += warn_count

        return cls(
            tool_name=tool_name,
            files=files,
            total_errors=total_errors,
            total_warnings=total_warnings,
            raw_output=raw_output,
        )

    def errors_by_path(self) -> dict[str, int]:
        """
        Return a simple mapping of file path to error count.

        Returns
        -------
        dict[str, int]
            File path to error count mapping.
        """
        return {path: diag.error_count for path, diag in self.files.items()}

    @staticmethod
    def empty(tool_name: str) -> DiagnosticReport:
        """
        Create an empty diagnostic report.

        Parameters
        ----------
        tool_name
            Name of the source tool.

        Returns
        -------
        DiagnosticReport
            Empty report with zero counts.
        """
        return DiagnosticReport(tool_name=tool_name, files={})


@dataclass(frozen=True)
class CoverageFileSummary:
    """Coverage summary for a single file.

    Attributes
    ----------
    rel_path
        Repository-relative file path.
    executed_lines
        Set of line numbers that were executed.
    missing_lines
        Set of executable line numbers that were not executed.
    """

    rel_path: str
    executed_lines: frozenset[int]
    missing_lines: frozenset[int]

    @property
    def total_executable(self) -> int:
        """Total number of executable lines."""
        return len(self.executed_lines) + len(self.missing_lines)

    @property
    def coverage_ratio(self) -> float:
        """Coverage ratio between 0.0 and 1.0."""
        total = self.total_executable
        if total == 0:
            return 1.0
        return len(self.executed_lines) / total


@dataclass(frozen=True)
class CoverageReport:
    """Aggregated coverage report from coverage.py.

    Attributes
    ----------
    files
        Sequence of per-file coverage summaries.
    total_executed
        Total executed lines across all files.
    total_missing
        Total missing lines across all files.
    json_path
        Path to the coverage JSON file, if generated.
    """

    files: Sequence[CoverageFileSummary]
    total_executed: int = 0
    total_missing: int = 0
    json_path: Path | None = None

    @classmethod
    def from_file_reports(
        cls,
        reports: Sequence[tuple[str, set[int], set[int]]],
        *,
        json_path: Path | None = None,
    ) -> CoverageReport:
        """
        Build a CoverageReport from raw file data.

        Parameters
        ----------
        reports
            Sequence of (rel_path, executed_lines, missing_lines) tuples.
        json_path
            Optional path to the generated JSON file.

        Returns
        -------
        CoverageReport
            Constructed report with aggregated totals.
        """
        files: list[CoverageFileSummary] = []
        total_executed = 0
        total_missing = 0

        for rel_path, executed, missing in reports:
            files.append(
                CoverageFileSummary(
                    rel_path=rel_path,
                    executed_lines=frozenset(executed),
                    missing_lines=frozenset(missing),
                )
            )
            total_executed += len(executed)
            total_missing += len(missing)

        return cls(
            files=tuple(files),
            total_executed=total_executed,
            total_missing=total_missing,
            json_path=json_path,
        )

    @staticmethod
    def empty() -> CoverageReport:
        """
        Create an empty coverage report.

        Returns
        -------
        CoverageReport
            Empty report with no files.
        """
        return CoverageReport(files=())

    def by_path(self) -> dict[str, CoverageFileSummary]:
        """
        Return a mapping of file path to summary.

        Returns
        -------
        dict[str, CoverageFileSummary]
            File path to summary mapping.
        """
        return {f.rel_path: f for f in self.files}


def parse_test_duration(entry: Mapping[str, object]) -> float:
    """Extract duration from test entry call data.

    Parameters
    ----------
    entry
        Test entry mapping.

    Returns
    -------
    float
        Duration in seconds.
    """
    call = entry.get("call")
    if isinstance(call, dict):
        dur_val = call.get("duration")
        if isinstance(dur_val, (int, float)):
            return float(dur_val)
    return 0.0


def parse_test_markers(entry: Mapping[str, object]) -> tuple[str, ...]:
    """Extract markers from test entry keywords.

    Parameters
    ----------
    entry
        Test entry mapping.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of marker names.
    """
    keywords = entry.get("keywords", {})
    if isinstance(keywords, dict):
        return tuple(sorted(k for k, v in keywords.items() if v))
    if isinstance(keywords, list):
        return tuple(sorted(str(k) for k in keywords))
    return ()


@dataclass(frozen=True)
class TestCaseResult:
    """Result for a single test case.

    Attributes
    ----------
    nodeid
        Pytest node identifier (e.g., "tests/test_app.py::test_foo").
    outcome
        Test outcome (passed, failed, skipped, error).
    duration_s
        Duration in seconds.
    markers
        List of pytest markers applied to this test.
    """

    nodeid: str
    outcome: str
    duration_s: float = 0.0
    markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestReport:
    """Aggregated test report from pytest.

    Attributes
    ----------
    tests
        Sequence of individual test results.
    passed_count
        Number of passed tests.
    failed_count
        Number of failed tests.
    skipped_count
        Number of skipped tests.
    error_count
        Number of tests with errors.
    total_duration_s
        Total test duration in seconds.
    report_path
        Path to the pytest JSON report file.
    """

    tests: Sequence[TestCaseResult]
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    total_duration_s: float = 0.0
    report_path: Path | None = None

    @classmethod
    def from_test_entries(
        cls,
        entries: Sequence[Mapping[str, object]],
        *,
        report_path: Path | None = None,
    ) -> TestReport:
        """
        Build a TestReport from raw pytest JSON entries.

        Parameters
        ----------
        entries
            Sequence of test entry mappings from pytest-json-report.
        report_path
            Optional path to the source report file.

        Returns
        -------
        TestReport
            Constructed report with aggregated counts.
        """
        tests: list[TestCaseResult] = []
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        total_duration = 0.0

        for entry in entries:
            nodeid = str(entry.get("nodeid", ""))
            if not nodeid:
                continue

            outcome = str(entry.get("outcome", entry.get("status", "unknown")))
            duration = parse_test_duration(entry)
            markers = parse_test_markers(entry)

            tests.append(
                TestCaseResult(
                    nodeid=nodeid,
                    outcome=outcome,
                    duration_s=duration,
                    markers=tuple(markers),
                )
            )

            total_duration += duration
            if outcome == "passed":
                passed += 1
            elif outcome == "failed":
                failed += 1
            elif outcome == "skipped":
                skipped += 1
            elif outcome == "error":
                errors += 1

        return cls(
            tests=tuple(tests),
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            error_count=errors,
            total_duration_s=total_duration,
            report_path=report_path,
        )

    @staticmethod
    def empty() -> TestReport:
        """
        Create an empty test report.

        Returns
        -------
        TestReport
            Empty report with no tests.
        """
        return TestReport(tests=())


@dataclass(frozen=True)
class ScipOccurrence:
    """A single SCIP symbol occurrence.

    Attributes
    ----------
    symbol
        SCIP symbol string.
    range_
        Line/column range as a tuple (start_line, start_col, end_line, end_col).
    symbol_roles
        Bitmask of symbol roles for this occurrence.
    """

    symbol: str
    range_: tuple[int, int, int, int]
    symbol_roles: int = 0

    @property
    def is_definition(self) -> bool:
        """Return True if the definition role is set."""
        return bool(self.symbol_roles & 1)

    def to_port_occurrence(self) -> port_types.ScipOccurrence:
        """Convert to port interface type.

        Returns
        -------
        port_types.ScipOccurrence
            Port-level occurrence with flattened range fields.
        """
        return port_types.ScipOccurrence(
            symbol=self.symbol,
            range_start_line=self.range_[0],
            range_start_col=self.range_[1],
            range_end_line=self.range_[2],
            range_end_col=self.range_[3],
            symbol_roles=self.symbol_roles,
        )


@dataclass(frozen=True)
class ScipDocument:
    """A single document in a SCIP index.

    Attributes
    ----------
    relative_path
        Repository-relative path to the document.
    symbols
        Symbols defined in this document.
    occurrences
        Sequence of symbol occurrences in this document.
    """

    relative_path: str
    symbols: Sequence[port_types.ScipSymbol] = ()
    occurrences: Sequence[ScipOccurrence] = ()

    def to_port_document(self) -> port_types.ScipDocument:
        """Convert to port interface type.

        Returns
        -------
        port_types.ScipDocument
            Port-level document with converted occurrences.
        """
        return port_types.ScipDocument(
            relative_path=self.relative_path,
            symbols=list(self.symbols),
            occurrences=[occ.to_port_occurrence() for occ in self.occurrences],
        )


@dataclass(frozen=True)
class ScipIndexResult:
    """Result from SCIP indexing.

    Attributes
    ----------
    documents
        Sequence of indexed documents.
    index_scip_path
        Path to the .scip binary index file.
    definition_count
        Total number of definitions across all documents.
    reference_count
        Total number of references across all documents.
    """

    documents: Sequence[ScipDocument]
    index_scip_path: Path | None = None
    definition_count: int = 0
    reference_count: int = 0

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[ScipDocument],
        *,
        index_scip_path: Path | None = None,
    ) -> ScipIndexResult:
        """Build a ScipIndexResult from parsed protobuf documents.

        Returns
        -------
        ScipIndexResult
            Aggregated result with definition/reference counts.
        """
        total_defs = 0
        total_refs = 0
        for doc in documents:
            for occ in doc.occurrences:
                if occ.is_definition:
                    total_defs += 1
                else:
                    total_refs += 1

        return cls(
            documents=tuple(documents),
            index_scip_path=index_scip_path,
            definition_count=total_defs,
            reference_count=total_refs,
        )

    @staticmethod
    def empty() -> ScipIndexResult:
        """
        Create an empty SCIP index result.

        Returns
        -------
        ScipIndexResult
            Empty result with no documents.
        """
        return ScipIndexResult(documents=())


ParsedToolResult = DiagnosticReport | CoverageReport | TestReport | ScipIndexResult


__all__ = [
    "CoverageFileSummary",
    "CoverageReport",
    "DiagnosticReport",
    "FileDiagnosticCount",
    "ParsedToolResult",
    "ReportProtocol",
    "ScipDocument",
    "ScipIndexResult",
    "ScipOccurrence",
    "TestCaseResult",
    "TestReport",
    "parse_test_duration",
    "parse_test_markers",
]
