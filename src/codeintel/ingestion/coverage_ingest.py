"""Backward compatibility shim for coverage ingestion.

This module provides the legacy `ingest_coverage_lines` function signature
for backward compatibility with existing code. New code should use
`CoverageIngestStep` from `codeintel.ingestion.steps.coverage_ingest`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.steps_coverage import CoverageIngestConfig
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def ingest_coverage_lines(
    gateway: StorageGateway,
    cfg: CoverageIngestConfig,
    *,
    tool_service: ToolService | None = None,
    tools: ToolsConfig | None = None,
) -> None:
    """
    Ingest coverage data from coverage.py into analytics.coverage_lines.

    This is a compatibility shim that wraps the new CoverageIngestStep.
    New code should use CoverageIngestStep directly.

    Parameters
    ----------
    gateway
        StorageGateway providing access to the target DuckDB database.
    cfg
        Coverage ingestion configuration (paths and identifiers).
    tool_service
        Optional ToolService for running coverage CLI.
    tools
        Optional ToolsConfig (unused, for backward compatibility).
    """
    from codeintel.ingestion.adapters import DuckDBStorageAdapter, ToolRunnerAdapter
    from codeintel.ingestion.steps.coverage_ingest import CoverageIngestStep

    _ = tools  # Unused in new implementation

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    tool = ToolRunnerAdapter(tool_service)

    # Execute step (async)
    step = CoverageIngestStep(storage=storage, tools=tool)

    # Get or create event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(
        step.execute_async(
            [],  # modules not used for coverage
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
            coverage_file=cfg.coverage_file,
        )
    )

    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "Unknown error"
        log.warning("Coverage ingest failed: %s", errors)


__all__ = ["ingest_coverage_lines"]
