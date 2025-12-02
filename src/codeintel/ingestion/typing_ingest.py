"""Typing analysis facade with convenient function-based API.

This module provides a function-based API for type annotation analysis
that wraps the class-based TypingIngestStep with sensible adapter defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.common import iter_modules
from codeintel.ingestion.steps.typing_ingest import AnnotationInfo, TypingIngestStep
from codeintel.storage.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.config import TypingIngestStepConfig
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway


def ingest_typing_signals(  # noqa: PLR0913
    gateway: StorageGateway,
    modules: Sequence[ModuleRecord] | None = None,
    *,
    cfg: TypingIngestStepConfig | None = None,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    code_profile: ScanProfile | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """Analyze type annotations and persist typedness metrics.

    This function provides a convenient entry point for type analysis
    that creates the necessary adapters and executes the step.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    modules
        Modules to analyze; if not provided, uses tracker modules.
    cfg
        Optional typing ingest step configuration (extracts repo/commit/repo_root).
    repo
        Repository identifier (overrides cfg.snapshot.repo if provided).
    commit
        Commit identifier (overrides cfg.snapshot.commit if provided).
    repo_root
        Repository root path (overrides cfg.snapshot.repo_root if provided).
    code_profile
        Optional scan profile (reserved for future use).
    tool_service
        Tool service for running external tools (currently unused).
    tracker
        Optional change tracker for incremental processing.

    Raises
    ------
    ValueError
        If neither cfg nor all of repo, commit, repo_root are provided.
    """
    # Reserved parameters for API compatibility
    del tool_service, code_profile

    # Resolve parameters from cfg or direct arguments
    if cfg is not None:
        actual_repo = repo or cfg.snapshot.repo
        actual_commit = commit or cfg.snapshot.commit
        actual_repo_root = repo_root or cfg.snapshot.repo_root
    else:
        actual_repo = repo
        actual_commit = commit
        actual_repo_root = repo_root

    # Validate required parameters
    if actual_repo is None or actual_commit is None or actual_repo_root is None:
        msg = "Must provide either cfg or all of repo, commit, repo_root"
        raise ValueError(msg)

    # Get modules from tracker, explicit list, or module inventory
    actual_modules: Sequence[ModuleRecord]
    if modules is not None:
        actual_modules = modules
    elif tracker is not None:
        actual_modules = tracker.modules
    else:
        # Load from module inventory
        module_map = load_module_map(
            gateway,
            actual_repo,
            actual_commit,
            language="python",
        )
        actual_modules = list(iter_modules(module_map, actual_repo_root))

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(actual_repo_root)

    # Create and execute step (sync version without external diagnostics)
    step = TypingIngestStep(storage=storage, discovery=discovery)
    step.execute(
        modules=actual_modules,
        repo=actual_repo,
        commit=actual_commit,
        repo_root=str(actual_repo_root),
    )


# Re-export for direct usage
__all__ = ["AnnotationInfo", "TypingIngestStep", "ingest_typing_signals"]
