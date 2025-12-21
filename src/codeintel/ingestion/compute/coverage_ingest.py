"""Coverage ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
test coverage data, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.compute.base import ExecutionResult
from codeintel.ingestion.row_serialization import row_serializer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)
COVERAGE_LINES_TABLE_KEY = "analytics.coverage_lines"


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
    ) -> ExecutionResult:
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
        ExecutionResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        result = await self._tools.run_coverage(
            repo_root,
            coverage_file=coverage_file,
        )

        if result.error is not None:
            log.warning("Coverage export failed: %s", result.error)
            return ExecutionResult.failed(f"Coverage export failed: {result.error}")

        all_rows: list[tuple[object, ...]] = []
        file_count = 0
        serializer = row_serializer_for_table_key(COVERAGE_LINES_TABLE_KEY)

        for file_data in result.files:
            file_count += 1
            rel_path = file_data.rel_path

            all_rows.extend(
                serializer(
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
                for line_num in file_data.executed_lines
            )

            all_rows.extend(
                serializer(
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
                for line_num in file_data.missing_lines
            )

        table_counts: dict[str, int] = {}

        if all_rows:
            scope = f"{repo}@{commit}"
            write_result = self._storage.write_batch(
                COVERAGE_LINES_TABLE_KEY,
                all_rows,
                scope=scope,
            )
            table_counts[COVERAGE_LINES_TABLE_KEY] = write_result.rows_affected

        log.info(
            "Coverage ingest: repo=%s commit=%s files=%d lines=%d",
            repo,
            commit,
            file_count,
            len(all_rows),
        )

        return ExecutionResult.ok(table_counts=table_counts)


__all__ = ["CoverageIngestStep"]
