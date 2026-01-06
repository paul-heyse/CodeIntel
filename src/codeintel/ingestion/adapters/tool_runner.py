"""Tool runner adapter implementing IngestToolPort.

This adapter wraps the existing ToolService to provide port-compliant
tool execution with normalized result types.

Architecture Note
-----------------
This adapter bridges two type systems:

**Input (ToolService side):**
    Rich "Report" types from ``tools/results.py`` with aggregated counts,
    factory methods, and helper functions (DiagnosticReport, TestReport,
    ScipIndexResult).

**Output (Port side):**
    Simpler "Result" types from ``ports/tools.py`` with status/error/duration
    fields (DiagnosticResult, TestResult, ScipResult).

The adapter calls ``ToolService`` methods, receives rich Report objects,
and converts them to the simpler Result types expected by the port interface.
This keeps domain logic rich while maintaining clean boundaries.

See Also
--------
codeintel.ingestion.tool_service : Service returning rich Report types
codeintel.ingestion.engine.results : Rich domain types (input)
codeintel.ingestion.ports.tools : Port interface types (output)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar

from codeintel.ingestion.ports.tools import (
    DiagnosticEntry,
    DiagnosticResult,
    ScipDocument,
    ScipOccurrence,
    ScipResult,
    ScipRunRequest,
    ScipSymbol,
    ToolStatus,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from codeintel.ingestion.engine.service import ToolService

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

    ADAPTER_NAME: ClassVar[str] = "tool_runner"

    def __init__(self, tool_service: ToolService) -> None:
        """Initialize the adapter with a tool service.

        Parameters
        ----------
        tool_service
            ToolService instance for executing tools.
        """
        self._service = tool_service

    def initialize(self) -> None:
        """Initialize the adapter (no-op, service is passed in constructor)."""

    def close(self) -> None:
        """Close the adapter (no-op, does not own service lifecycle)."""

    @property
    def is_available(self) -> bool:
        """Check if adapter is available.

        Returns
        -------
        bool
            True if tool service is available.
        """
        return self._service is not None

    async def _run_diagnostic_tool(
        self,
        tool_name: str,
        repo_root: Path,
        *,
        paths: Sequence[Path] | None = None,
    ) -> DiagnosticResult:
        """Run a diagnostic tool and convert to DiagnosticResult.

        Parameters
        ----------
        tool_name
            Name of the tool (for logging and diagnostic entries).
            Must correspond to a method name on the service (run_{tool_name}).
        repo_root
            Repository root directory.
        paths
            Optional paths to scope diagnostics (relative to repo_root or absolute).

        Returns
        -------
        DiagnosticResult
            Diagnostic results with error counts per file.
        """
        service_method: Callable[..., Awaitable[dict[str, int]]] = getattr(
            self._service,
            f"run_{tool_name}",
        )
        start = time.perf_counter()
        try:
            errors_by_path = await service_method(repo_root, paths=paths)
            duration = time.perf_counter() - start

            diagnostics = [
                DiagnosticEntry(
                    path=path,
                    line=1,
                    column=1,
                    severity="error",
                    code=tool_name,
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
            log.warning("%s execution failed: %s", tool_name, exc)
            return DiagnosticResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )

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
        return await self._run_diagnostic_tool("pyright", repo_root, paths=paths)

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
        return await self._run_diagnostic_tool("pyrefly", repo_root, paths=paths)

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
        return await self._run_diagnostic_tool("ruff", repo_root, paths=paths)

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
        start = time.perf_counter()
        try:
            scip_result = await self._service.run_scip_full(
                request,
            )

            duration = time.perf_counter() - start

            documents = _convert_scip_documents(scip_result.documents or [])

            scip_path = scip_result.index_scip_path or request.output_scip
            scip_exists = _check_file_exists(scip_path)

            return ScipResult(
                status=ToolStatus.OK,
                documents=documents,
                index_scip_path=scip_path if scip_exists else None,
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
            symbols=[
                ScipSymbol(symbol=sym.symbol, documentation=sym.documentation)
                for sym in (getattr(doc, "symbols", None) or [])
            ],
            occurrences=[
                _convert_scip_occurrence(
                    occ,
                    position_encoding=getattr(doc, "position_encoding", None),
                    text_document_encoding=getattr(doc, "text_document_encoding", None),
                )
                for occ in (getattr(doc, "occurrences", None) or [])
            ],
            position_encoding=getattr(doc, "position_encoding", None),
            text_document_encoding=getattr(doc, "text_document_encoding", None),
        )
        for doc in documents
    ]


def _convert_scip_occurrence(
    occ: object,
    *,
    position_encoding: int | None,
    text_document_encoding: str | None,
) -> ScipOccurrence:
    """Convert a single SCIP occurrence to port type.

    Parameters
    ----------
    occ
        Occurrence from SCIP service.
    position_encoding
        Override position encoding from the parent document if provided.
    text_document_encoding
        Override text document encoding from the SCIP metadata if provided.

    Returns
    -------
    ScipOccurrence
        Converted occurrence.
    """
    occ_range = getattr(occ, "range_", None)
    if isinstance(occ_range, tuple):
        start_line, start_col, end_line, end_col = occ_range
    else:
        raw_range = getattr(occ, "range", None) or []
        start_line = raw_range[_RANGE_START_LINE] if raw_range else 0
        start_col = raw_range[_RANGE_START_COL] if len(raw_range) >= _MIN_RANGE_LEN_COL else 0
        end_line = (
            raw_range[_RANGE_END_LINE] if len(raw_range) >= _MIN_RANGE_LEN_END_LINE else start_line
        )
        end_col = (
            raw_range[_RANGE_END_COL] if len(raw_range) >= _MIN_RANGE_LEN_END_COL else start_col
        )

    occ_position_encoding = getattr(occ, "position_encoding", None)
    occ_text_encoding = getattr(occ, "text_document_encoding", None)

    return ScipOccurrence(
        symbol=getattr(occ, "symbol", ""),
        range_start_line=start_line,
        range_start_col=start_col,
        range_end_line=end_line,
        range_end_col=end_col,
        symbol_roles=getattr(occ, "symbol_roles", 0) or 0,
        position_encoding=(
            occ_position_encoding if occ_position_encoding is not None else position_encoding
        ),
        text_document_encoding=(
            occ_text_encoding if occ_text_encoding is not None else text_document_encoding
        ),
        start_byte=getattr(occ, "start_byte", None),
        end_byte=getattr(occ, "end_byte", None),
    )
