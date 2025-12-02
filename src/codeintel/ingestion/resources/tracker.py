"""Tracker provider for lazy access to change tracker.

This module provides `TrackerProvider`, a resource provider that
manages access to the change tracker for incremental ingestion.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.ingestion.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.plugins.protocol import IngestRuntimeScratch
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuration for change tracker creation.

    Encapsulate optional parameters for TrackerProvider to reduce
    constructor argument count.

    Attributes
    ----------
    scratch
        Shared scratch space (may contain tracker from repo_scan).
    profile
        Optional scan profile for filtering.
    policy
        Optional incremental ingest policy.
    full_rebuild
        Whether to force full rebuild mode.
    """

    scratch: IngestRuntimeScratch | None = field(default=None)
    profile: ScanProfile | None = field(default=None)
    policy: IncrementalIngestPolicy | None = field(default=None)
    full_rebuild: bool = field(default=False)


class TrackerProvider(LazyResource["ChangeTracker"]):
    """Lazy provider for change tracker.

    Load or build a change tracker for incremental ingestion.
    The tracker can come from:
    1. A preloaded tracker (set via set_preloaded)
    2. The scratch store (populated by repo_scan plugin)
    3. Created fresh from the gateway and snapshot

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    config
        Optional tracker configuration with scratch, profile, policy settings.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        config: TrackerConfig | None = None,
    ) -> None:
        """Initialize the tracker provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        config
            Optional tracker configuration with scratch, profile, policy settings.
        """
        super().__init__("TrackerProvider")
        self._gateway = gateway
        self._snapshot = snapshot
        self._config = config or TrackerConfig()

    @property
    def _scratch(self) -> IngestRuntimeScratch | None:
        """Access scratch from config for backward compatibility."""
        return self._config.scratch

    @property
    def _profile(self) -> ScanProfile | None:
        """Access profile from config for backward compatibility."""
        return self._config.profile

    @property
    def _policy(self) -> IncrementalIngestPolicy | None:
        """Access policy from config for backward compatibility."""
        return self._config.policy

    @property
    def _full_rebuild(self) -> bool:
        """Access full_rebuild from config for backward compatibility."""
        return self._config.full_rebuild

    def _load(self) -> ChangeTracker:
        """Load or build change tracker.

        Returns
        -------
        ChangeTracker
            The change tracker.
        """
        from codeintel.ingestion.change_tracker import ChangeTracker

        # First check scratch store (populated by repo_scan)
        if self._scratch is not None:
            tracker = self._scratch.consume("change_tracker")
            if tracker is not None and isinstance(tracker, ChangeTracker):
                log.debug(
                    "Loaded tracker from scratch: repo=%s commit=%s",
                    self._snapshot.repo,
                    self._snapshot.commit,
                )
                return tracker

        # Create fresh tracker
        log.debug(
            "Creating fresh tracker: repo=%s commit=%s",
            self._snapshot.repo,
            self._snapshot.commit,
        )

        from codeintel.ingestion.adapters import (
            DuckDBStorageAdapter,
            FilesystemDiscoveryAdapter,
            HashChangeDetectionAdapter,
        )
        from codeintel.ingestion.infrastructure_utilities.source_scanner import (
            default_code_profile,
        )
        from codeintel.ingestion.ports.change_detection import ChangeRequest
        from codeintel.ingestion.steps.repo_scan import RepoScanStep

        # Create adapters
        storage = DuckDBStorageAdapter(self._gateway)
        discovery = FilesystemDiscoveryAdapter(self._snapshot.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)

        # Use provided profile or default
        actual_profile = self._profile or default_code_profile(self._snapshot.repo_root)

        # Run repo scan to build tracker
        step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
        )
        _result, modules, _change_set = step.execute(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            repo_root=self._snapshot.repo_root,
            profile=actual_profile,
            full_rebuild=self._full_rebuild,
        )

        # Build change request
        change_request = ChangeRequest(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            repo_root=self._snapshot.repo_root,
            language="python",
            full_rebuild=self._full_rebuild,
            scan_profile=actual_profile,
        )

        # Create and return tracker
        return ChangeTracker.create(
            gateway=self._gateway,
            change_request=change_request,
            modules=modules,
            policy=self._policy,
            change_detection=change_detection,
        )

    def get_or_create(self) -> ChangeTracker:
        """Get or create the change tracker.

        Alias for get() that makes the intent clearer.

        Returns
        -------
        ChangeTracker
            The change tracker.
        """
        return self.get()


__all__ = ["TrackerConfig", "TrackerProvider"]
