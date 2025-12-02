"""Repository scanning facade with convenient function-based API.

This module provides a function-based API for repository scanning that
wraps the class-based RepoScanStep with sensible adapter defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.infrastructure_utilities.source_scanner import default_code_profile
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.steps.repo_scan import RepoScanStep

if TYPE_CHECKING:
    from codeintel.config import RepoScanStepConfig
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.storage.gateway import StorageGateway


def ingest_repo(
    gateway: StorageGateway,
    cfg: RepoScanStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    apply_schema: bool = True,
) -> ChangeTracker:
    """Scan repository and return a change tracker.

    This function provides a convenient entry point for repository scanning
    that creates the necessary adapters and executes the step.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    cfg
        Repository scan step configuration.
    code_profile
        Optional scan profile for filtering modules.
    apply_schema
        Whether to apply schema (ignored, schema applied by gateway).

    Returns
    -------
    ChangeTracker
        Change tracker populated with discovered modules.
    """
    # apply_schema kept for backward compatibility; schema applied by gateway
    del apply_schema

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(cfg.snapshot.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)

    # Use provided profile or default
    actual_profile = code_profile or default_code_profile(cfg.snapshot.repo_root)

    # Create and execute step
    step = RepoScanStep(
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
    )
    _result, modules, _change_set = step.execute(
        repo=cfg.snapshot.repo,
        commit=cfg.snapshot.commit,
        repo_root=cfg.snapshot.repo_root,
        profile=actual_profile,
        full_rebuild=False,
    )

    # Build change request for the tracker
    change_request = ChangeRequest(
        repo=cfg.snapshot.repo,
        commit=cfg.snapshot.commit,
        repo_root=cfg.snapshot.repo_root,
        language="python",
        full_rebuild=False,
        scan_profile=actual_profile,
    )

    # Build and return change tracker using factory method
    return ChangeTracker.create(
        gateway=gateway,
        change_request=change_request,
        modules=modules,
        policy=None,
        change_detection=change_detection,
    )


# Re-export step class for direct usage
__all__ = ["RepoScanStep", "ingest_repo"]
