"""Tool adapter bridging build protocols to ingestion ports.

This adapter implements IngestToolPort using minimal protocol shapes that match
the build providers, allowing the ingestion compute layer to remain decoupled
from the build execution engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from codeintel.ingestion.ports.tools import (
    CoverageFileData,
    CoverageResult,
    DiagnosticEntry,
    DiagnosticResult,
    ScipResult,
    ToolStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from codeintel.build.types import CoverageData, ScipIndexResult, TypeCheckResult

log = logging.getLogger(__name__)


class TypeChecker(Protocol):
    """Minimal type-checking protocol used by ingestion tooling."""

    async def check(
        self,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
        config_path: Path | None = None,
    ) -> TypeCheckResult: ...


class CoverageCollector(Protocol):
    """Minimal coverage collection protocol used by ingestion tooling."""

    async def collect(self, coverage_file: Path) -> Mapping[str, CoverageData]: ...


class ScipIndexer(Protocol):
    """Minimal SCIP indexing protocol used by ingestion tooling."""

    async def index(
        self,
        repo_root: Path,
        output_path: Path,
        *,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> ScipIndexResult: ...


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
    """

    def __init__(
        self,
        *,
        type_checker: TypeChecker | None = None,
        coverage_collector: CoverageCollector | None = None,
        scip_indexer: ScipIndexer | None = None,
    ) -> None:
        self._type_checker = type_checker
        self._coverage_collector = coverage_collector
        self._scip_indexer = scip_indexer

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
                    column=d.character,
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
        _ = self, repo_root

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
        _ = output_json, target_dir, rel_paths
        if self._scip_indexer is None:
            return ScipResult(
                status=ToolStatus.SKIPPED,
                error="SCIP indexer not available",
            )
        try:
            result = await self._scip_indexer.index(repo_root, output_scip)
            if not result.success:
                return ScipResult(
                    status=ToolStatus.FAILED,
                    error=result.error_message or "SCIP indexing failed",
                )

            return ScipResult(
                status=ToolStatus.OK,
                documents=[],
                index_scip_path=result.index_path,
                index_json_path=None,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ScipResult(
                status=ToolStatus.FAILED,
                error=str(exc),
            )


__all__ = ["BuildToolAdapter"]
