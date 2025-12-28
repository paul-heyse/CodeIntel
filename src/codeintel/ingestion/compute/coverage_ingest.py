"""Coverage ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
test coverage data, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)
COVERAGE_LINES_TABLE_KEY = "analytics.coverage_lines"


class CoverageIngestStep:
    """Coverage ingestion step with port injection.

    This step ingests test coverage data from coverage.py,
    using ports for all I/O operations.

    Parameters
    ----------
    tools
        Tool port for running coverage export.
    """

    def __init__(
        self,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        tools
            Tool port for running coverage.
        """
        self._tools = tools

    async def execute_async(
        self,
        _modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        coverage_file: Path | None = None,
    ) -> CoverageIngestResult:
        """Execute coverage ingestion.

        Parameters
        ----------
        _modules
            Modules for reference (coverage is file-based, not used directly).
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.
        coverage_file
            Optional explicit coverage data file path.

        Returns
        -------
        CoverageIngestResult
            Result bundle with row tuples and execution status.
        """
        created_at = datetime.now(UTC)

        result = await self._tools.run_coverage(
            repo_root,
            coverage_file=coverage_file,
        )

        if result.error is not None:
            log.warning("Coverage export failed: %s", result.error)
            return CoverageIngestResult(
                result=ExecutionResult.failed(f"Coverage export failed: {result.error}")
            )

        buffer = columnar_buffer_for_table_key(COVERAGE_LINES_TABLE_KEY)
        file_count = 0

        for file_data in result.files:
            file_count += 1
            rel_path = file_data.rel_path

            for line_num in file_data.executed_lines:
                buffer.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "rel_path": rel_path,
                        "line": line_num,
                        "is_executable": True,
                        "is_covered": True,
                        "hits": 1,
                        "context_count": 0,
                        "created_at": created_at,
                    }
                )
            for line_num in file_data.missing_lines:
                buffer.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "rel_path": rel_path,
                        "line": line_num,
                        "is_executable": True,
                        "is_covered": False,
                        "hits": 0,
                        "context_count": 0,
                        "created_at": created_at,
                    }
                )

        log.info(
            "Coverage ingest: repo=%s commit=%s files=%d lines=%d",
            repo,
            commit,
            file_count,
            buffer.row_count,
        )

        return CoverageIngestResult(
            result=ExecutionResult.ok(),
            rows=buffer.data,
            row_count=buffer.row_count,
        )


@dataclass(frozen=True)
class CoverageIngestResult:
    """Result bundle for coverage ingestion."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    row_count: int = 0


__all__ = ["CoverageIngestResult", "CoverageIngestStep"]
