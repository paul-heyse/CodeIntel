"""Session-scoped manifest caching for build operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "BuildSession",
]


@dataclass
class BuildSession:
    """Session-scoped caches and state for a build run.

    Provides caching for loaded manifests (keyed by target name).

    Attributes
    ----------
    snapshot
        The repository snapshot for this build session.
    gateway
        Storage gateway for database access.

    Examples
    --------
    >>> session = BuildSession(snapshot, gateway)
    >>> session.preload_manifests()
    """

    snapshot: SnapshotRef
    gateway: StorageGateway
    _manifest_cache: dict[str, OutputManifest] = field(default_factory=dict, repr=False)
    _manifests_preloaded: bool = field(default=False, repr=False)

    def get_manifest(self, target_name: str) -> OutputManifest | None:
        """Return cached manifest or load and cache.

        Parameters
        ----------
        target_name
            Name of the target to load manifest for.

        Returns
        -------
        OutputManifest | None
            The manifest if found, None otherwise.
        """
        if target_name in self._manifest_cache:
            return self._manifest_cache[target_name]

        manifest = self.gateway.build.load_manifest(
            target=target_name,
            repo=self.snapshot.repo,
            commit=self.snapshot.commit,
        )
        if manifest is not None:
            self._manifest_cache[target_name] = manifest
        return manifest

    def preload_manifests(self) -> None:
        """Bulk-load all manifests for the snapshot.

        This is more efficient than loading manifests one-by-one when
        computing state for many targets. After preloading, get_manifest
        returns from cache without DB queries.
        """
        if self._manifests_preloaded:
            return

        manifests = self.gateway.build.list_manifests(
            repo=self.snapshot.repo,
            commit=self.snapshot.commit,
        )
        self._manifest_cache = {m.target: m for m in manifests}
        self._manifests_preloaded = True

    def cached_manifest_targets(self) -> frozenset[str]:
        """Return the set of target names present in the manifest cache.

        Returns
        -------
        frozenset[str]
            Target names for which a manifest is cached.
        """
        return frozenset(self._manifest_cache)

    def seed_manifest_cache(self, manifests: Mapping[str, OutputManifest]) -> None:
        """Seed the manifest cache with externally provided entries.

        This method merges the provided mapping into the existing cache without
        marking the cache as fully preloaded. This avoids incorrectly treating
        missing entries as absent from storage when the provided mapping is
        partial.

        Parameters
        ----------
        manifests
            Mapping of target name to manifest to cache.
        """
        for name, manifest in manifests.items():
            self._manifest_cache.setdefault(name, manifest)

    def invalidate_manifest(self, target_name: str) -> None:
        """Invalidate cached manifest for a target.

        Use after a target is computed to force re-loading from DB.

        Parameters
        ----------
        target_name
            Name of the target to invalidate.
        """
        self._manifest_cache.pop(target_name, None)

    def clear_caches(self) -> None:
        """Clear all caches.

        Use when starting a new computation pass where cached values
        may be stale.
        """
        self._hash_cache.clear()
        self._manifest_cache.clear()
        self._manifests_preloaded = False

    @property
    def cached_manifest_count(self) -> int:
        """Return the number of cached manifests.

        Returns
        -------
        int
            Count of manifests in cache.
        """
        return len(self._manifest_cache)

    @property
    def cached_hash_count(self) -> int:
        """Return the number of cached hashes.

        Returns
        -------
        int
            Count of hashes in cache.
        """
        return len(self._hash_cache)
