"""Module provider for lazy loading of module lists.

This module provides `ModuleProvider`, a resource provider that
lazily loads module records from either a change tracker or
the module inventory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter
from codeintel.ingestion.resources.protocol import LazyResource
from codeintel.storage.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.utilities.scanning import ScanProfile
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class ModuleProvider(LazyResource[Sequence["ModuleRecord"]]):
    """Lazy provider for module records.

    Load module records from either a change tracker (if available)
    or the module inventory.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    tracker
        Optional change tracker (if available, modules come from tracker).
    profile
        Optional scan profile for filtering.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        tracker: ChangeTracker | None = None,
        profile: ScanProfile | None = None,
    ) -> None:
        """Initialize the module provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        tracker
            Optional change tracker (if available, modules come from tracker).
        profile
            Optional scan profile for filtering.
        """
        super().__init__("ModuleProvider")
        self._gateway = gateway
        self._snapshot = snapshot
        self._tracker = tracker
        self._profile = profile

    def _load(self) -> Sequence[ModuleRecord]:
        """Load module records.

        Returns
        -------
        Sequence[ModuleRecord]
            List of module records.
        """
        # If tracker available, use its modules
        if self._tracker is not None:
            log.debug(
                "Loading modules from tracker: repo=%s commit=%s count=%d",
                self._snapshot.repo,
                self._snapshot.commit,
                len(self._tracker.modules),
            )
            return self._tracker.modules

        # Otherwise load from module inventory
        module_map = load_module_map(
            self._gateway,
            self._snapshot.repo,
            self._snapshot.commit,
            language="python",
            logger=log,
        )

        modules = list(
            FilesystemDiscoveryAdapter.iter_modules(
                module_map,
                self._snapshot.repo_root,
                logger=log,
                scan_profile=self._profile,
            )
        )

        log.debug(
            "Loaded modules from inventory: repo=%s commit=%s count=%d",
            self._snapshot.repo,
            self._snapshot.commit,
            len(modules),
        )
        return modules

    @property
    def tracker(self) -> ChangeTracker | None:
        """Return the underlying tracker if available.

        Returns
        -------
        ChangeTracker | None
            The change tracker or None.
        """
        return self._tracker

    def with_tracker(self, tracker: ChangeTracker) -> ModuleProvider:
        """Create a new provider with a tracker.

        Parameters
        ----------
        tracker
            Change tracker to use.

        Returns
        -------
        ModuleProvider
            New provider instance with tracker.
        """
        return ModuleProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            tracker=tracker,
            profile=self._profile,
        )


__all__ = ["ModuleProvider"]
