"""Coverage ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
test coverage data, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.steps.base import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)


class CoverageIngestStep:
    """Coverage ingestion step with port injection.

    This step ingests test coverage data from coverage.py,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    tools
        Tool port for running coverage export.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        tools
            Tool port for running coverage.
        """
        self._storage = storage
        self._tools = tools

    async def execute_async(
        self,
        _modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        coverage_file: Path | None = None,
    ) -> StepResult:
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
        StepResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        # Run coverage export
        result = await self._tools.run_coverage(
            repo_root,
            coverage_file=coverage_file,
        )

        if result.error is not None:
            log.warning("Coverage export failed: %s", result.error)
            return StepResult.fail(f"Coverage export failed: {result.error}")

        # Build rows using comprehension
        all_rows: list[list[object]] = [
            [
                repo,
                commit,
                file_data.rel_path,
                len(file_data.executed_lines),
                len(file_data.missing_lines),
                sorted(file_data.executed_lines),
                sorted(file_data.missing_lines),
                sorted(file_data.excluded_lines),
                created_at,
            ]
            for file_data in result.files
        ]

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if all_rows:
            scope = f"{repo}@{commit}"
            write_result = self._storage.write_batch("core.coverage", all_rows, scope=scope)
            table_counts["core.coverage"] = write_result.rows_written
            total_rows = write_result.rows_written

        log.info(
            "Coverage ingest: repo=%s commit=%s files=%d",
            repo,
            commit,
            len(all_rows),
        )

        return StepResult(
            rows_written=total_rows,
            table_counts=table_counts,
        )


__all__ = ["CoverageIngestStep"]
