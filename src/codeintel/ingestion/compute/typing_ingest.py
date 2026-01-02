"""Typing diagnostics ingestion step with port injection.

This module provides a pure domain logic implementation for collecting
static diagnostics, using ports for all I/O operations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsStaticDiagnosticsRow as StaticDiagnosticRow,
)
from codeintel.ingestion.ports.tools import ToolStatus

DIAGNOSTICS_TABLE_KEY = "analytics.static_diagnostics"

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)


@dataclass
class DiagnosticCounts:
    """Error counts from diagnostic tools.

    Attributes
    ----------
    pyright
        Errors from pyright.
    pyrefly
        Errors from pyrefly.
    ruff
        Errors from ruff.
    """

    pyright: dict[str, int]
    pyrefly: dict[str, int]
    ruff: dict[str, int]


async def _collect_diagnostic_counts(
    repo_root: Path,
    tools: IngestToolPort,
) -> DiagnosticCounts:
    """Collect error counts from all diagnostic tools.

    Parameters
    ----------
    repo_root
        Repository root directory.
    tools
        Tool port for running diagnostics.

    Returns
    -------
    DiagnosticCounts
        Error counts from each tool.
    """
    pyright_result = await tools.run_pyright(repo_root)
    pyrefly_result = await tools.run_pyrefly(repo_root)
    ruff_result = await tools.run_ruff(repo_root)

    return DiagnosticCounts(
        pyright=pyright_result.errors_by_path() if pyright_result.status == ToolStatus.OK else {},
        pyrefly=pyrefly_result.errors_by_path() if pyrefly_result.status == ToolStatus.OK else {},
        ruff=ruff_result.errors_by_path() if ruff_result.status == ToolStatus.OK else {},
    )


class TypingIngestStep:
    """Typing diagnostics ingestion step with port injection.

    This step collects static diagnostics using the tool ports.

    Parameters
    ----------
    tools
        Optional tool port for running diagnostics.
    """

    def __init__(
        self,
        tools: IngestToolPort | None = None,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        tools
            Optional tool port for running diagnostics.
        """
        self._tools = tools

    async def execute_async(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: str,
        run_diagnostics: bool = True,
    ) -> TypingIngestResult:
        """Execute typing analysis on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.
        run_diagnostics
            Whether to run external diagnostic tools.

        Returns
        -------
        TypingIngestResult
            Result bundle with row tuples and execution status.
        """
        diag_counts = DiagnosticCounts(pyright={}, pyrefly={}, ruff={})
        if run_diagnostics and self._tools is not None:
            diag_counts = await _collect_diagnostic_counts(Path(repo_root), self._tools)

        diagnostic_buffer = self._process_modules(modules, repo, commit, diag_counts)

        log.info(
            "Typing ingest: repo=%s commit=%s diagnostics=%d",
            repo,
            commit,
            diagnostic_buffer.row_count,
        )

        return TypingIngestResult(
            result=ExecutionResult.ok(),
            diagnostic_rows=diagnostic_buffer.data,
            diagnostic_row_count=diagnostic_buffer.row_count,
        )

    def _process_modules(
        self,
        modules: Sequence[ModuleRecord],
        repo: str,
        commit: str,
        diag_counts: DiagnosticCounts,
    ) -> ColumnarRowBuffer:
        """Process modules and build diagnostic rows.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        diag_counts
            Diagnostic counts from tools.

        Returns
        -------
        ColumnarRowBuffer
            Diagnostic rows.
        """
        diagnostic_buffer = columnar_buffer_for_table_key(DIAGNOSTICS_TABLE_KEY)

        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue

            pyright_errors = diag_counts.pyright.get(module.rel_path, 0)
            pyrefly_errors = diag_counts.pyrefly.get(module.rel_path, 0)
            ruff_errors = diag_counts.ruff.get(module.rel_path, 0)
            type_error_count = pyright_errors + pyrefly_errors + ruff_errors

            diagnostic_buffer.append(
                StaticDiagnosticRow(
                    repo=repo,
                    commit=commit,
                    rel_path=module.rel_path,
                    pyright_errors=pyright_errors,
                    pyrefly_errors=pyrefly_errors,
                    ruff_errors=ruff_errors,
                    total_errors=type_error_count,
                    has_errors=type_error_count > 0,
                )
            )

        return diagnostic_buffer

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: str,
    ) -> TypingIngestResult:
        """Execute typing analysis synchronously (without diagnostics).

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.

        Returns
        -------
        TypingIngestResult
            Result bundle with row tuples and execution status.
        """
        return asyncio.run(
            self.execute_async(
                modules, repo=repo, commit=commit, repo_root=repo_root, run_diagnostics=False
            )
        )


@dataclass(frozen=True)
class TypingIngestResult:
    """Result bundle for typing ingestion."""

    result: ExecutionResult
    diagnostic_rows: ColumnarRows = field(default_factory=dict)
    diagnostic_row_count: int = 0


__all__ = [
    "TypingIngestResult",
    "TypingIngestStep",
]
