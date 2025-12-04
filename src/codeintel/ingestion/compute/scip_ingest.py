"""SCIP indexing step with port injection.

This module provides a pure domain logic implementation for running
SCIP indexing and ingesting symbol data, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.ports.tools import ToolStatus

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort, ScipDocument

log = logging.getLogger(__name__)

# Threshold for incremental vs full indexing
INCREMENTAL_INDEX_THRESHOLD = 100


@dataclass(frozen=True)
class ScipIngestResult:
    """Outcome of SCIP ingestion.

    Attributes
    ----------
    status
        Result status: "success", "unavailable", or "failed".
    index_scip
        Path to SCIP index file if created.
    index_json
        Path to JSON export if created.
    reason
        Reason for failure or unavailability.
    """

    status: Literal["success", "unavailable", "failed"]
    index_scip: Path | None
    index_json: Path | None
    reason: str | None = None


@dataclass
class ScipIngestConfig:
    """Configuration for SCIP indexing.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    repo_root
        Repository root path.
    output_scip
        Path for SCIP index output.
    output_json
        Path for JSON export output.
    target_dir
        Optional target directory to index.
    """

    repo: str
    commit: str
    repo_root: Path
    output_scip: Path
    output_json: Path
    target_dir: Path | None = None


class ScipIngestStep:
    """SCIP indexing step with port injection.

    This step runs SCIP indexing and persists symbol data,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    tools
        Tool port for running SCIP.
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
            Tool port for running SCIP.
        """
        self._storage = storage
        self._tools = tools

    async def execute_async(
        self,
        modules: Sequence[ModuleRecord],
        config: ScipIngestConfig,
    ) -> StepResult:
        """Execute SCIP indexing.

        Parameters
        ----------
        modules
            Modules to index (can be subset for incremental).
        config
            SCIP indexing configuration.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        # Determine if doing incremental (specific files) or full index
        rel_paths = None
        if len(modules) < INCREMENTAL_INDEX_THRESHOLD:
            rel_paths = [m.rel_path for m in modules if m.rel_path.endswith(".py")]

        # Run SCIP indexing
        result = await self._tools.run_scip(
            config.repo_root,
            output_scip=config.output_scip,
            output_json=config.output_json,
            target_dir=config.target_dir,
            rel_paths=rel_paths,
        )

        if result.status != ToolStatus.OK:
            log.warning("SCIP indexing failed: %s", result.error)
            return StepResult.fail(f"SCIP indexing failed: {result.error}")

        # Build rows
        symbol_rows = _build_symbol_rows(result.documents, config.repo, config.commit, created_at)
        occurrence_rows = _build_occurrence_rows(
            result.documents, config.repo, config.commit, created_at
        )

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if symbol_rows:
            scope = f"{config.repo}@{config.commit}"
            write_result = self._storage.write_batch("core.scip_symbols", symbol_rows, scope=scope)
            table_counts["core.scip_symbols"] = write_result.rows_written
            total_rows += write_result.rows_written

        if occurrence_rows:
            write_result = self._storage.write_batch("core.scip_occurrences", occurrence_rows)
            table_counts["core.scip_occurrences"] = write_result.rows_written
            total_rows += write_result.rows_written

        log.info(
            "SCIP ingest: repo=%s commit=%s symbols=%d occurrences=%d",
            config.repo,
            config.commit,
            len(symbol_rows),
            len(occurrence_rows),
        )

        return StepResult(rows_written=total_rows, table_counts=table_counts)


def _build_symbol_rows(
    documents: list[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
) -> list[list[object]]:
    """Build symbol rows from SCIP documents.

    Parameters
    ----------
    documents
        SCIP documents containing symbols.
    repo
        Repository identifier.
    commit
        Commit identifier.
    created_at
        Row creation timestamp.

    Returns
    -------
    list[list[object]]
        Symbol rows for storage.
    """
    rows: list[list[object]] = []
    for doc in documents:
        rows.extend(
            [repo, commit, doc.relative_path, sym.symbol, sym.documentation, created_at]
            for sym in doc.symbols
        )
    return rows


def _build_occurrence_rows(
    documents: list[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
) -> list[list[object]]:
    """Build occurrence rows from SCIP documents.

    Parameters
    ----------
    documents
        SCIP documents containing occurrences.
    repo
        Repository identifier.
    commit
        Commit identifier.
    created_at
        Row creation timestamp.

    Returns
    -------
    list[list[object]]
        Occurrence rows for storage.
    """
    rows: list[list[object]] = []
    for doc in documents:
        rows.extend(
            [
                repo,
                commit,
                doc.relative_path,
                occ.symbol,
                occ.range_start_line,
                occ.range_start_col,
                occ.range_end_line,
                occ.range_end_col,
                occ.symbol_roles,
                created_at,
            ]
            for occ in doc.occurrences
        )
    return rows


__all__ = ["ScipIngestConfig", "ScipIngestResult", "ScipIngestStep"]
