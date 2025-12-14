"""Domain result types for tool plugin outputs.

This module defines structured result types that tool plugins return after
parsing raw tool output. These domain objects provide type safety and a
clear contract between tool plugins and ingestion consumers.

Architecture Note
-----------------
These "Report" types (DiagnosticReport, CoverageReport, TestReport, ScipIndexResult)
are **rich domain objects** used internally by tool plugins. They include:
- Aggregated counts (total_errors, total_warnings, definition_count, etc.)
- Factory methods for construction from raw data (from_error_counts, from_json_documents)
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
    from collections.abc import Mapping, Sequence
    from pathlib import Path

MIN_SCIP_RANGE_FIELDS = 3
FULL_SCIP_RANGE_FIELDS = 4


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
    is_definition
        Whether this occurrence is a definition.
    """

    symbol: str
    range_: tuple[int, int, int, int]
    is_definition: bool = False

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
            symbol_roles=1 if self.is_definition else 0,
        )


@dataclass(frozen=True)
class ScipDocument:
    """A single document in a SCIP index.

    Attributes
    ----------
    relative_path
        Repository-relative path to the document.
    occurrences
        Sequence of symbol occurrences in this document.
    """

    relative_path: str
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
            symbols=[],  # Domain type doesn't track symbols separately
            occurrences=[occ.to_port_occurrence() for occ in self.occurrences],
        )


def parse_scip_range(rng: Sequence[object]) -> tuple[int, int, int, int] | None:
    """Parse SCIP range from list to tuple.

    SCIP ranges have 3 or 4 elements. Three-element ranges represent
    occurrences on a single line with start_col and end_col.

    Parameters
    ----------
    rng
        Range sequence from SCIP output.

    Returns
    -------
    tuple[int, int, int, int] | None
        Normalized (start_line, start_col, end_line, end_col) or None.
    """
    try:
        int_values = [int(x) for x in rng if isinstance(x, (int, float, str))]
    except (ValueError, TypeError):
        return None
    if len(int_values) != len(rng):
        return None
    if len(int_values) == MIN_SCIP_RANGE_FIELDS:
        return (int_values[0], int_values[1], int_values[0], int_values[2])
    if len(int_values) == FULL_SCIP_RANGE_FIELDS:
        return (int_values[0], int_values[1], int_values[2], int_values[3])
    return None


def parse_scip_occurrence(occ: Mapping[str, object]) -> tuple[ScipOccurrence, bool] | None:
    """Parse a single SCIP occurrence from a dict.

    Parameters
    ----------
    occ
        Occurrence dict from SCIP JSON.

    Returns
    -------
    tuple[ScipOccurrence, bool] | None
        Tuple of (occurrence, is_definition) or None if invalid.
    """
    symbol = occ.get("symbol")
    if not isinstance(symbol, str):
        return None

    rng = occ.get("range", [])
    if not isinstance(rng, list) or len(rng) < MIN_SCIP_RANGE_FIELDS:
        return None

    range_tuple = parse_scip_range(rng)
    if range_tuple is None:
        return None

    roles = occ.get("symbol_roles", 0)
    is_def = bool(roles & 1) if isinstance(roles, int) else False

    return (
        ScipOccurrence(symbol=symbol, range_=range_tuple, is_definition=is_def),
        is_def,
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
    index_json_path
        Path to the JSON export of the index.
    definition_count
        Total number of definitions across all documents.
    reference_count
        Total number of references across all documents.
    """

    documents: Sequence[ScipDocument]
    index_scip_path: Path | None = None
    index_json_path: Path | None = None
    definition_count: int = 0
    reference_count: int = 0

    @classmethod
    def from_json_documents(
        cls,
        docs: Sequence[Mapping[str, object]],
        *,
        index_scip_path: Path | None = None,
        index_json_path: Path | None = None,
    ) -> ScipIndexResult:
        """Build a ScipIndexResult from parsed JSON documents.

        Parameters
        ----------
        docs
            Sequence of document mappings from SCIP JSON export.
        index_scip_path
            Optional path to the .scip binary file.
        index_json_path
            Optional path to the JSON export.

        Returns
        -------
        ScipIndexResult
            Constructed result with aggregated counts.
        """
        documents: list[ScipDocument] = []
        total_defs = 0
        total_refs = 0

        for doc in docs:
            rel_path = doc.get("relative_path")
            if not isinstance(rel_path, str):
                continue

            occurrences_raw = doc.get("occurrences", [])
            occurrences: list[ScipOccurrence] = []

            if isinstance(occurrences_raw, list):
                for occ in occurrences_raw:
                    if not isinstance(occ, dict):
                        continue
                    parsed = parse_scip_occurrence(occ)
                    if parsed is None:
                        continue
                    occurrence, is_def = parsed
                    occurrences.append(occurrence)
                    if is_def:
                        total_defs += 1
                    else:
                        total_refs += 1

            documents.append(ScipDocument(relative_path=rel_path, occurrences=tuple(occurrences)))

        return cls(
            documents=tuple(documents),
            index_scip_path=index_scip_path,
            index_json_path=index_json_path,
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
    "parse_scip_occurrence",
    "parse_scip_range",
    "parse_test_duration",
    "parse_test_markers",
]
