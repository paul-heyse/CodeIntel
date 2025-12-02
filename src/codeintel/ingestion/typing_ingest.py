"""Backward compatibility shim for typing signals ingestion.

This module provides the legacy `ingest_typing_signals` function signature
for backward compatibility with existing code. New code should use
`TypingIngestStep` from `codeintel.ingestion.steps.typing_ingest`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.steps_typing import TypingIngestConfig
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestConfig,
    *,
    tool_service: ToolService | None = None,
) -> None:
    """
    Analyze typing signals and populate analytics.typedness + static_diagnostics.

    This is a compatibility shim that wraps the new TypingIngestStep.
    New code should use TypingIngestStep directly.

    Parameters
    ----------
    gateway
        StorageGateway providing access to the target DuckDB database.
    cfg
        Typing ingestion configuration.
    tool_service
        Optional ToolService for running pyright/pyrefly.
    """
    from codeintel.ingestion.adapters import (
        DuckDBStorageAdapter,
        FilesystemDiscoveryAdapter,
        ToolRunnerAdapter,
    )
    from codeintel.ingestion.steps.typing_ingest import TypingIngestStep

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(cfg.repo_root)
    tools = ToolRunnerAdapter(tool_service)

    # Execute step (async)
    step = TypingIngestStep(storage=storage, discovery=discovery, tools=tools)

    # Get or create event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(
        step.execute_async(
            [],  # modules - will be discovered
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=str(cfg.repo_root),
        )
    )

    if not result.success:
        errors = "; ".join(result.errors) if result.errors else "Unknown error"
        log.warning("Typing ingest failed: %s", errors)


__all__ = ["ingest_typing_signals"]
