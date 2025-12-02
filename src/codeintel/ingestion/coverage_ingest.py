"""Coverage data ingestion facade with convenient function-based API.

This module provides a function-based API for coverage data ingestion
that wraps the class-based CoverageIngestStep with sensible adapter defaults.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters import DuckDBStorageAdapter, ToolRunnerAdapter
from codeintel.ingestion.steps.coverage_ingest import CoverageIngestStep

if TYPE_CHECKING:
    from codeintel.config import CoverageIngestStepConfig
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway


def ingest_coverage_lines(  # noqa: PLR0913
    gateway: StorageGateway,
    modules: Sequence[ModuleRecord] | None = None,
    *,
    cfg: CoverageIngestStepConfig | None = None,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    coverage_file: Path | None = None,
    tool_service: ToolService | None = None,
    json_output_path: Path | None = None,
    tracker: ChangeTracker | None = None,
    tools: ToolsConfig | None = None,
) -> None:
    """Ingest coverage data and persist coverage lines.

    This function provides a convenient entry point for coverage ingestion
    that creates the necessary adapters and executes the step.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    modules
        Modules to process; if not provided, uses tracker modules.
    cfg
        Optional coverage ingest step configuration.
    repo
        Repository identifier; defaults to config value.
    commit
        Commit identifier; defaults to config value.
    repo_root
        Repository root path; defaults to config value.
    coverage_file
        Path to coverage data file.
    tool_service
        Tool service for running external tools.
    json_output_path
        Path to write JSON output (reserved for future use).
    tracker
        Optional change tracker for incremental processing.
    tools
        Tools configuration (reserved for future use).

    Raises
    ------
    ValueError
        If required parameters are not provided.
    """
    # Reserved parameters for API compatibility
    del tools, json_output_path

    # Resolve parameters from config or direct args
    if cfg is not None:
        actual_repo = repo or cfg.snapshot.repo
        actual_commit = commit or cfg.snapshot.commit
        actual_repo_root = repo_root or cfg.snapshot.repo_root
        actual_coverage_file = coverage_file or getattr(cfg, "coverage_file", None)
    else:
        actual_repo = repo
        actual_commit = commit
        actual_repo_root = repo_root
        actual_coverage_file = coverage_file

    # Validate required params
    if actual_repo is None or actual_commit is None or actual_repo_root is None:
        msg = "Must provide either cfg or all of repo, commit, repo_root"
        raise ValueError(msg)

    # Get modules from tracker if not provided
    actual_modules: Sequence[ModuleRecord]
    if modules is not None:
        actual_modules = modules
    elif tracker is not None:
        actual_modules = tracker.modules
    else:
        actual_modules = []

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)

    # Create tool adapter if tool_service provided
    if tool_service is not None:
        tools_adapter = ToolRunnerAdapter(tool_service)
        step = CoverageIngestStep(storage=storage, tools=tools_adapter)

        # Run async step
        asyncio.run(
            step.execute_async(
                _modules=actual_modules,
                repo=actual_repo,
                commit=actual_commit,
                repo_root=actual_repo_root,
                coverage_file=actual_coverage_file,
            )
        )


# Re-export step class for direct usage
__all__ = ["CoverageIngestStep", "ingest_coverage_lines"]
