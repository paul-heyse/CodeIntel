"""Tool adapter bridging build protocols to ingestion ports.

This adapter implements IngestToolPort using the protocols from
codeintel.build.protocols, allowing the ingestion compute layer
to work with the new build system's tool abstractions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.tools import (
    CoverageFileData,
    CoverageResult,
    DiagnosticEntry,
    DiagnosticResult,
    ScipResult,
    TestCase,
    TestResult,
    ToolStatus,
)

if TYPE_CHECKING:
    from codeintel.build.protocols import (
        CoverageCollector,
        ScipIndexer,
        TestReporter,
        TypeChecker,
    )

log = logging.getLogger(__name__)


class BuildToolAdapter:
    """Adapter from build protocols to ingestion tool port.

    This adapter wraps the build system's protocol-based tool providers
    and exposes them as an IngestToolPort-compatible interface.

    Parameters
    ----------
    type_checker
        Type checker protocol implementation (optional).
    coverage_collector
        Coverage collector protocol implementation (optional).
    scip_indexer
        SCIP indexer protocol implementation (optional).
    test_reporter
        Test reporter protocol implementation (optional).
    """

    def __init__(
        self,
        *,
        type_checker: TypeChecker | None = None,
        coverage_collector: CoverageCollector | None = None,
        scip_indexer: ScipIndexer | None = None,
        test_reporter: TestReporter | None = None,
    ) -> None:
        self._type_checker = type_checker
        self._coverage_collector = coverage_collector
        self._scip_indexer = scip_indexer
        self._test_reporter = test_reporter

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
        if self._type_checker is None:
            return DiagnosticResult(
                status=ToolStatus.SKIPPED,
                error="Type checker not available",
            )
        try:
            result = await self._type_checker.check(repo_root)
            diagnostics = [
                DiagnosticEntry(
                    path=d.path,
                    line=d.line,
                    column=d.character,  # TypeDiagnostic uses character
                    severity=d.severity,
                    code=d.code or "pyright",
                    message=d.message,
                )
                for d in result.diagnostics
            ]
            return DiagnosticResult(
                status=ToolStatus.OK if result.success else ToolStatus.FAILED,
                diagnostics=diagnostics,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return DiagnosticResult(
                status=ToolStatus.FAILED,
                error=str(exc),
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
        # Pyrefly uses the same interface as pyright
        return await self.run_pyright(repo_root)

    async def run_ruff(self, repo_root: Path) -> DiagnosticResult:
        """Run ruff linter.

        Parameters
        ----------
        repo_root
            Repository root directory (unused, included for interface compatibility).

        Returns
        -------
        DiagnosticResult
            Linting results - always returns SKIPPED.
        """
        # Mark self and repo_root as used for interface compatibility
        _ = self, repo_root
        # Ruff would need its own protocol - for now return skipped
        return DiagnosticResult(
            status=ToolStatus.SKIPPED,
            error="Ruff linting not available via build adapter",
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
            Repository root directory (unused, included for interface compatibility).
        coverage_file
            Optional explicit coverage data file path.
        output_path
            Optional path for JSON output (unused, included for interface compatibility).

        Returns
        -------
        CoverageResult
            Coverage data for all files.
        """
        # Mark unused parameters for interface compatibility
        _ = repo_root, output_path
        if self._coverage_collector is None:
            return CoverageResult(
                status=ToolStatus.SKIPPED,
                error="Coverage collector not available",
            )
        if coverage_file is None:
            return CoverageResult(
                status=ToolStatus.FAILED,
                error="Coverage file path not provided",
            )
        try:
            # CoverageCollector.collect returns Mapping[str, CoverageData]
            result = await self._coverage_collector.collect(coverage_file)
            files = [
                CoverageFileData(
                    rel_path=path,
                    executed_lines=data.covered_lines,
                    missing_lines=data.missing_lines,
                )
                for path, data in result.items()
            ]
            return CoverageResult(
                status=ToolStatus.OK,
                files=files,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return CoverageResult(
                status=ToolStatus.FAILED,
                error=str(exc),
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
            Path for JSON export output (unused, SCIP outputs single file).
        target_dir
            Optional target directory to index (unused, included for interface compatibility).
        rel_paths
            Optional list of specific files to index (unused, included for interface compatibility).

        Returns
        -------
        ScipResult
            SCIP indexing results.
        """
        # Mark unused parameters for interface compatibility
        _ = output_json, target_dir, rel_paths
        if self._scip_indexer is None:
            return ScipResult(
                status=ToolStatus.SKIPPED,
                error="SCIP indexer not available",
            )
        try:
            # ScipIndexer.index uses output_path for the single output file
            result = await self._scip_indexer.index(repo_root, output_scip)
            if not result.success:
                return ScipResult(
                    status=ToolStatus.FAILED,
                    error=result.error_message or "SCIP indexing failed",
                )
            # ScipIndexResult only contains success/path, not parsed documents
            # Document parsing happens separately in the compute layer
            return ScipResult(
                status=ToolStatus.OK,
                documents=[],  # Documents not provided by simple indexer
                index_scip_path=result.index_path,
                index_json_path=None,  # JSON not generated via this adapter
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ScipResult(
                status=ToolStatus.FAILED,
                error=str(exc),
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
            Repository root directory (unused, included for interface compatibility).
        json_report_path
            Path for JSON report output.

        Returns
        -------
        TestResult
            Test execution results.
        """
        # Mark unused parameters for interface compatibility
        _ = repo_root
        if self._test_reporter is None:
            return TestResult(
                status=ToolStatus.SKIPPED,
                error="Test reporter not available",
            )
        try:
            # TestReporter.collect returns tuple[TestResult, ...]
            results = await self._test_reporter.collect(json_report_path)
            tests = [
                TestCase(
                    nodeid=t.node_id,
                    outcome=t.outcome,
                    duration_s=t.duration_ms / 1000.0,  # Convert ms to seconds
                )
                for t in results
            ]
            passed = sum(1 for t in results if t.outcome == "passed")
            failed = sum(1 for t in results if t.outcome == "failed")
            skipped = sum(1 for t in results if t.outcome == "skipped")
            return TestResult(
                status=ToolStatus.OK,
                tests=tests,
                passed=passed,
                failed=failed,
                skipped=skipped,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return TestResult(
                status=ToolStatus.FAILED,
                error=str(exc),
            )


__all__ = ["BuildToolAdapter"]
