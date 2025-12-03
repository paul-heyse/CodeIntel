"""Tooling port protocol for external tool runners.

This module defines the ToolingPort protocol that abstracts external tool
execution. The production adapter uses ToolRunner/ToolService from codeintel.

Note
----
Tests should use the existing infrastructure in tests/_helpers/tooling.py
which wraps the production ToolRunner and ToolService classes. This protocol
exists to formalize the interface for dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """Result from running an external tool.

    Attributes
    ----------
    success : bool
        Whether the tool completed successfully.
    stdout : str
        Standard output from tool.
    stderr : str
        Standard error from tool.
    exit_code : int
        Tool exit code.
    artifacts : dict[str, Path]
        Paths to generated artifacts (e.g., coverage.json).
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class CoverageResult:
    """Result from running coverage collection.

    Attributes
    ----------
    success : bool
        Whether coverage collection succeeded.
    coverage_file : Path | None
        Path to coverage data file if successful.
    report : dict[str, object]
        Parsed coverage report data.
    """

    success: bool
    coverage_file: Path | None
    report: dict[str, object]


@runtime_checkable
class ToolingPort(Protocol):
    """Protocol for external tool execution.

    Defines the interface for running external tools. The production
    implementation uses ToolRunner from codeintel.ingestion.

    See tests/_helpers/tooling.py for the real implementation using
    ToolRunner and ToolService.
    """

    def run_coverage(
        self,
        target: Path,
        source_dirs: list[Path],
        *,
        parallel: bool = False,
    ) -> CoverageResult:
        """Run coverage collection on a target.

        Parameters
        ----------
        target
            Script or test file to run with coverage.
        source_dirs
            Directories to include in coverage measurement.
        parallel
            Whether to run in parallel mode.

        Returns
        -------
        CoverageResult
            Coverage collection result with data path.
        """
        ...

    def run_pyright(
        self,
        target: Path,
        *,
        python_version: str = "3.13",
    ) -> ToolResult:
        """Run pyright type checker on a target.

        Parameters
        ----------
        target
            File or directory to type check.
        python_version
            Python version for type checking.

        Returns
        -------
        ToolResult
            Type check result with diagnostics.
        """
        ...

    def run_scip_python(
        self,
        repo_root: Path,
        output_path: Path,
    ) -> ToolResult:
        """Run scip-python indexer on a repository.

        Parameters
        ----------
        repo_root
            Repository root to index.
        output_path
            Path for output SCIP index.

        Returns
        -------
        ToolResult
            Indexing result with artifact path.
        """
        ...


__all__ = [
    "CoverageResult",
    "ToolResult",
    "ToolingPort",
]
