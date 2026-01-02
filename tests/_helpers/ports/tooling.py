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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


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
        Paths to generated artifacts (e.g., reports).
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    artifacts: dict[str, Path]


@runtime_checkable
class ToolingPort(Protocol):
    """Protocol for external tool execution.

    Defines the interface for running external tools. The production
    implementation uses ToolRunner from codeintel.ingestion.

    See tests/_helpers/tooling.py for the real implementation using
    ToolRunner and ToolService.
    """

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
    "ToolResult",
    "ToolingPort",
]
