"""Configuration ingestion facade with convenient function-based API.

This module provides a function-based API for configuration file ingestion
that wraps the class-based ConfigIngestStep with sensible adapter defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.steps.config_ingest import ConfigIngestStep

if TYPE_CHECKING:
    from codeintel.config import ConfigIngestStepConfig
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway


def ingest_config_values(
    gateway: StorageGateway,
    cfg: ConfigIngestStepConfig,
    *,
    config_files: Sequence[ModuleRecord] | None = None,
    config_profile: ScanProfile | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """Ingest configuration files and persist flattened values.

    This function provides a convenient entry point for config ingestion
    that creates the necessary adapters and executes the step.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    cfg
        Configuration ingest step configuration.
    config_files
        Configuration files to process.
    config_profile
        Optional scan profile for filtering config files (reserved for future use).
    tracker
        Optional change tracker for incremental processing.
    """
    # config_profile reserved for future use
    del config_profile

    # Get config files from tracker if not provided
    actual_files: Sequence[ModuleRecord]
    if config_files is not None:
        actual_files = config_files
    elif tracker is not None:
        # Filter modules that look like config files
        actual_files = [
            m
            for m in tracker.modules
            if m.rel_path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"))
        ]
    else:
        actual_files = []

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(cfg.snapshot.repo_root)

    # Create and execute step
    step = ConfigIngestStep(storage=storage, discovery=discovery)
    step.execute(
        config_files=actual_files,
        repo=cfg.snapshot.repo,
        commit=cfg.snapshot.commit,
    )


# Re-export step class for direct usage
__all__ = ["ConfigIngestStep", "ingest_config_values"]
