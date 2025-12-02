"""Backward compatibility shim for repository scanning.

This module provides the legacy `ingest_repo` function signature
for backward compatibility with existing code. New code should use
`RepoScanStep` from `codeintel.ingestion.steps.repo_scan`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.steps_repo import RepoScanConfig
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def ingest_repo(
    gateway: StorageGateway,
    cfg: RepoScanConfig,
) -> ChangeTracker:
    """
    Scan repository modules and return a ChangeTracker.

    This is a compatibility shim that wraps the new RepoScanStep.
    New code should use RepoScanStep directly.

    Parameters
    ----------
    gateway
        StorageGateway providing access to the target DuckDB database.
    cfg
        Repository scan configuration.

    Returns
    -------
    ChangeTracker
        Change tracker with module data.
    """
    from codeintel.ingestion.adapters import (
        DuckDBStorageAdapter,
        FilesystemDiscoveryAdapter,
        HashChangeDetectionAdapter,
    )
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ports.change_detection import ChangeRequest
    from codeintel.ingestion.steps.repo_scan import RepoScanStep

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(cfg.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)

    # Execute step
    step = RepoScanStep(
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
    )

    _result, modules, _change_set = step.execute(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        profile=cfg.code_profile,
        full_rebuild=cfg.full_rebuild if hasattr(cfg, "full_rebuild") else False,
    )

    # Build change request for tracker
    change_request = ChangeRequest(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        language="python",
        full_rebuild=cfg.full_rebuild if hasattr(cfg, "full_rebuild") else False,
        scan_profile=cfg.code_profile,
    )

    # Create change tracker
    return ChangeTracker.create(
        gateway=gateway,
        change_request=change_request,
        modules=modules,
        policy=None,
        change_detection=change_detection,
    )


__all__ = ["ingest_repo"]
