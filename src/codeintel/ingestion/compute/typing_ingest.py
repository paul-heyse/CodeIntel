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
    empty_table_for_table,
    table_for_columnar_rows,
)
from codeintel.ingestion.ports.tools import DiagnosticResult, ToolStatus

DIAGNOSTICS_TABLE_KEY = "analytics.static_diagnostics"

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    import pyarrow as pa

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


@dataclass(frozen=True)
class TypingIngestContext:
    """Context inputs for typing diagnostics ingestion.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    repo_root
        Repository root path.
    scope_paths
        Optional paths to scope diagnostics (relative to repo_root or absolute).
    run_diagnostics
        Whether to run external diagnostic tools.
    """

    repo: str
    commit: str
    repo_root: str
    scope_paths: Sequence[Path] | None = None
    run_diagnostics: bool = True


async def _collect_diagnostic_counts(
    repo_root: Path,
    tools: IngestToolPort,
    *,
    scope_paths: Sequence[Path] | None = None,
) -> DiagnosticCounts:
    """Collect error counts from all diagnostic tools.

    Parameters
    ----------
    repo_root
        Repository root directory.
    tools
        Tool port for running diagnostics.
    scope_paths
        Optional paths to scope diagnostics (relative to repo_root or absolute).

    Returns
    -------
    DiagnosticCounts
        Error counts from each tool.
    """

    async def _run_tool(
        label: str,
        coro: Awaitable[DiagnosticResult],
    ) -> DiagnosticResult:
        try:
            return await coro
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("typing diagnostics %s failed: %s", label, exc)
            return DiagnosticResult(status=ToolStatus.FAILED, error=str(exc))

    pyright_task = _run_tool("pyright", tools.run_pyright(repo_root, paths=scope_paths))
    pyrefly_task = _run_tool("pyrefly", tools.run_pyrefly(repo_root, paths=scope_paths))
    ruff_task = _run_tool("ruff", tools.run_ruff(repo_root, paths=scope_paths))

    pyright_result, pyrefly_result, ruff_result = await asyncio.gather(
        pyright_task,
        pyrefly_task,
        ruff_task,
    )

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
        context: TypingIngestContext,
    ) -> TypingIngestResult:
        """Execute typing analysis on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        context
            Typing ingestion context for repository metadata and scope.

        Returns
        -------
        TypingIngestResult
            Result bundle with row tuples and execution status.
        """
        diag_counts = DiagnosticCounts(pyright={}, pyrefly={}, ruff={})
        if context.run_diagnostics and self._tools is not None:
            diag_counts = await _collect_diagnostic_counts(
                Path(context.repo_root),
                self._tools,
                scope_paths=context.scope_paths,
            )

        diagnostic_buffer = self._process_modules(
            modules,
            context.repo,
            context.commit,
            diag_counts,
        )

        log.info(
            "Typing ingest: repo=%s commit=%s diagnostics=%d",
            context.repo,
            context.commit,
            diagnostic_buffer.row_count,
        )

        diagnostic_rows_reader, row_count = table_for_columnar_rows(
            DIAGNOSTICS_TABLE_KEY,
            diagnostic_buffer.data,
            extras_policy="retain",
        )
        return TypingIngestResult(
            result=ExecutionResult.ok(),
            diagnostic_rows=diagnostic_buffer.data,
            diagnostic_rows_reader=diagnostic_rows_reader,
            diagnostic_row_count=row_count,
        )

    @staticmethod
    def _process_modules(
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
                {
                    "repo": repo,
                    "commit": commit,
                    "rel_path": module.rel_path,
                    "pyright_errors": pyright_errors,
                    "pyrefly_errors": pyrefly_errors,
                    "ruff_errors": ruff_errors,
                    "total_errors": type_error_count,
                    "has_errors": type_error_count > 0,
                }
            )

        return diagnostic_buffer

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        context: TypingIngestContext,
    ) -> TypingIngestResult:
        """Execute typing analysis synchronously (without diagnostics).

        Parameters
        ----------
        modules
            Modules to process.
        context
            Typing ingestion context for repository metadata and scope.

        Returns
        -------
        TypingIngestResult
            Result bundle with row tuples and execution status.
        """
        return asyncio.run(
            self.execute_async(
                modules,
                context=TypingIngestContext(
                    repo=context.repo,
                    commit=context.commit,
                    repo_root=context.repo_root,
                    scope_paths=context.scope_paths,
                    run_diagnostics=False,
                ),
            )
        )


@dataclass(frozen=True)
class TypingIngestResult:
    """Result bundle for typing ingestion."""

    result: ExecutionResult
    diagnostic_rows: ColumnarRows = field(default_factory=dict)
    diagnostic_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(DIAGNOSTICS_TABLE_KEY)
    )
    diagnostic_row_count: int = 0


__all__ = [
    "TypingIngestContext",
    "TypingIngestResult",
    "TypingIngestStep",
]
