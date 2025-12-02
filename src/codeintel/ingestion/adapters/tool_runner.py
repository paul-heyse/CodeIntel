"""Tool runner adapter implementing IngestToolPort.

This adapter wraps the existing ToolService to provide port-compliant
tool execution with normalized result types.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.tools import (
    CoverageFileData,
    CoverageResult,
    DiagnosticEntry,
    DiagnosticResult,
    ScipDocument,
    ScipOccurrence,
    ScipResult,
    ScipSymbol,
    TestCase,
    TestResult,
    ToolStatus,
)

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService

log = logging.getLogger(__name__)


def _check_file_exists(path: Path) -> bool:
    """Check if a file exists (sync helper for async context).

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
    bool
        True if file exists.
    """
    return path.is_file()


# Constants for SCIP range indices
_RANGE_START_LINE = 0
_RANGE_START_COL = 1
_RANGE_END_LINE = 2
_RANGE_END_COL = 3
_MIN_RANGE_LEN_COL = 2
_MIN_RANGE_LEN_END_LINE = 3
_MIN_RANGE_LEN_END_COL = 4


class ToolRunnerAdapter:
    """Tool runner adapter implementing IngestToolPort.

    This adapter wraps the existing ToolService to provide a port-compliant
    interface with normalized result types.

    Parameters
    ----------
    tool_service
        ToolService instance for executing tools.
    """

    def __init__(self, tool_service: ToolService) -> None:
        """Initialize the adapter with a tool service.

        Parameters
        ----------
        tool_service
            ToolService instance for executing tools.
        """
        self._service = tool_service

    @classmethod
    def from_runner(cls, runner: ToolRunner, tools_config: ToolsConfig) -> ToolRunnerAdapter:
        """Create adapter from a ToolRunner.

        Parameters
        ----------
        runner
            ToolRunner for executing external tools.
        tools_config
            Tools configuration.

        Returns
        -------
        ToolRunnerAdapter
            Configured adapter instance.
        """
        from codeintel.ingestion.tool_service import ToolService

        service = ToolService(runner, tools_config)
        return cls(service)

    async def run_pyright(self, repo_root: Path) -> DiagnosticResult:
        """Run pyright type checker.

        Parameters
        ----------
        repo_root
            Repository root directory.

        Returns
        -------
        DiagnosticResult
            Type checking results with diagnostics.
        """
        start = time.perf_counter()
        try:
            errors_by_path = await self._service.run_pyright(repo_root)
            duration = time.perf_counter() - start

            diagnostics = [
                DiagnosticEntry(
                    path=path,
                    line=1,
                    column=1,
                    severity="error",
                    code="pyright",
                    message=f"{count} error(s)",
                )
                for path, count in errors_by_path.items()
                if count > 0
            ]

            return DiagnosticResult(
                status=ToolStatus.OK,
                diagnostics=diagnostics,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("pyright execution failed: %s", exc)
            return DiagnosticResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

    async def run_pyrefly(self, repo_root: Path) -> DiagnosticResult:
        """Run pyrefly type checker.

        Parameters
        ----------
        repo_root
            Repository root directory.

        Returns
        -------
        DiagnosticResult
            Type checking results with diagnostics.
        """
        start = time.perf_counter()
        try:
            errors_by_path = await self._service.run_pyrefly(repo_root)
            duration = time.perf_counter() - start

            diagnostics = [
                DiagnosticEntry(
                    path=path,
                    line=1,
                    column=1,
                    severity="error",
                    code="pyrefly",
                    message=f"{count} error(s)",
                )
                for path, count in errors_by_path.items()
                if count > 0
            ]

            return DiagnosticResult(
                status=ToolStatus.OK,
                diagnostics=diagnostics,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("pyrefly execution failed: %s", exc)
            return DiagnosticResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

    async def run_ruff(self, repo_root: Path) -> DiagnosticResult:
        """Run ruff linter.

        Parameters
        ----------
        repo_root
            Repository root directory.

        Returns
        -------
        DiagnosticResult
            Linting results with diagnostics.
        """
        start = time.perf_counter()
        try:
            errors_by_path = await self._service.run_ruff(repo_root)
            duration = time.perf_counter() - start

            diagnostics = [
                DiagnosticEntry(
                    path=path,
                    line=1,
                    column=1,
                    severity="error",
                    code="ruff",
                    message=f"{count} error(s)",
                )
                for path, count in errors_by_path.items()
                if count > 0
            ]

            return DiagnosticResult(
                status=ToolStatus.OK,
                diagnostics=diagnostics,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("ruff execution failed: %s", exc)
            return DiagnosticResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

    async def run_coverage(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageResult:
        """Run coverage tool to export coverage data.

        Parameters
        ----------
        repo_root
            Repository root directory.
        coverage_file
            Optional explicit coverage data file path.
        output_path
            Optional path for JSON output.

        Returns
        -------
        CoverageResult
            Coverage data for all files.
        """
        start = time.perf_counter()
        try:
            reports = await self._service.run_coverage_json(
                repo_root,
                coverage_file=coverage_file,
                output_path=output_path,
            )
            duration = time.perf_counter() - start

            files = [
                CoverageFileData(
                    rel_path=report.rel_path,
                    executed_lines=frozenset(report.executed_lines),
                    missing_lines=frozenset(report.missing_lines),
                )
                for report in reports
            ]

            return CoverageResult(
                status=ToolStatus.OK,
                files=files,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("coverage execution failed: %s", exc)
            return CoverageResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

    async def run_scip(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
        rel_paths: list[str] | None = None,
    ) -> ScipResult:
        """Run SCIP indexing.

        Parameters
        ----------
        repo_root
            Repository root directory.
        output_scip
            Path for SCIP index output.
        output_json
            Path for JSON export output.
        target_dir
            Optional target directory to index.
        rel_paths
            Optional list of specific files to index.

        Returns
        -------
        ScipResult
            SCIP indexing results.
        """
        start = time.perf_counter()
        try:
            if rel_paths is not None:
                scip_result = await self._service.run_scip_shard(
                    repo_root,
                    rel_paths=rel_paths,
                    output_scip=output_scip,
                    output_json=output_json,
                    target_dir=target_dir,
                )
            else:
                scip_result = await self._service.run_scip_full(
                    repo_root,
                    output_scip=output_scip,
                    output_json=output_json,
                    target_dir=target_dir,
                )

            duration = time.perf_counter() - start

            documents = _convert_scip_documents(scip_result.documents or [])

            # Check file existence - stored as variables to avoid async path check
            scip_exists = _check_file_exists(output_scip)
            json_exists = _check_file_exists(output_json)

            return ScipResult(
                status=ToolStatus.OK,
                documents=documents,
                index_scip_path=output_scip if scip_exists else None,
                index_json_path=output_json if json_exists else None,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("SCIP execution failed: %s", exc)
            return ScipResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

    async def run_pytest(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> TestResult:
        """Run pytest and generate JSON report.

        Parameters
        ----------
        repo_root
            Repository root directory.
        json_report_path
            Path for JSON report output.

        Returns
        -------
        TestResult
            Test execution results.
        """
        start = time.perf_counter()
        try:
            await self._service.run_pytest_report(repo_root, json_report_path=json_report_path)
            duration = time.perf_counter() - start

            # Parse the JSON report if available (sync file read)
            if _check_file_exists(json_report_path):
                return _parse_pytest_report(json_report_path, duration)

            return TestResult(
                status=ToolStatus.OK,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            duration = time.perf_counter() - start
            log.warning("pytest execution failed: %s", exc)
            return TestResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )


__all__ = ["ToolRunnerAdapter"]


def _convert_scip_documents(documents: Sequence[object]) -> list[ScipDocument]:
    """Convert SCIP service documents to port types.

    Parameters
    ----------
    documents
        Documents from SCIP service.

    Returns
    -------
    list[ScipDocument]
        Converted documents.
    """
    return [
        ScipDocument(
            relative_path=getattr(doc, "relative_path", ""),
            symbols=tuple(
                ScipSymbol(symbol=sym.symbol, documentation=sym.documentation)
                for sym in (getattr(doc, "symbols", None) or [])
            ),
            occurrences=tuple(
                _convert_scip_occurrence(occ) for occ in (getattr(doc, "occurrences", None) or [])
            ),
        )
        for doc in documents
    ]


def _convert_scip_occurrence(occ: object) -> ScipOccurrence:
    """Convert a single SCIP occurrence to port type.

    Parameters
    ----------
    occ
        Occurrence from SCIP service.

    Returns
    -------
    ScipOccurrence
        Converted occurrence.
    """
    occ_range = getattr(occ, "range", None) or []

    start_line = occ_range[_RANGE_START_LINE] if occ_range else 0
    start_col = occ_range[_RANGE_START_COL] if len(occ_range) >= _MIN_RANGE_LEN_COL else 0
    end_line = (
        occ_range[_RANGE_END_LINE] if len(occ_range) >= _MIN_RANGE_LEN_END_LINE else start_line
    )
    end_col = occ_range[_RANGE_END_COL] if len(occ_range) >= _MIN_RANGE_LEN_END_COL else start_col

    return ScipOccurrence(
        symbol=getattr(occ, "symbol", ""),
        range_start_line=start_line,
        range_start_col=start_col,
        range_end_line=end_line,
        range_end_col=end_col,
        symbol_roles=getattr(occ, "symbol_roles", 0) or 0,
    )


def _parse_pytest_report(json_report_path: Path, duration: float) -> TestResult:
    """Parse pytest JSON report.

    Parameters
    ----------
    json_report_path
        Path to JSON report.
    duration
        Total execution duration.

    Returns
    -------
    TestResult
        Parsed test results.
    """
    data = json.loads(json_report_path.read_text(encoding="utf-8"))
    tests = [
        TestCase(
            nodeid=test.get("nodeid", ""),
            outcome=test.get("outcome", "unknown"),
            duration_s=test.get("duration", 0.0),
            longrepr=test.get("longrepr"),
        )
        for test in data.get("tests", [])
    ]
    summary = data.get("summary", {})
    return TestResult(
        status=ToolStatus.OK,
        tests=tests,
        passed=summary.get("passed", 0),
        failed=summary.get("failed", 0),
        skipped=summary.get("skipped", 0),
        duration_s=duration,
    )
