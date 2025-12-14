"""Base data loader implementation.

This module provides BaseDataLoader, an abstract base class for
data loaders with built-in caching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from codeintel.core.data.snapshot import SnapshotKey, SnapshotScopedCache

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


class BaseDataLoader[T](ABC):
    """Abstract base class for data loaders with caching.

    Subclasses implement `_load_impl()` to define the actual
    loading logic. The base class handles caching automatically.

    Examples
    --------
    >>> class FunctionLoader(BaseDataLoader[list[dict]]):
    ...     def _load_impl(
    ...         self,
    ...         gateway: StorageGateway,
    ...         *,
    ...         repo: str,
    ...         commit: str,
    ...     ) -> list[dict]:
    ...         # Load from database
    ...         return []
    """

    def __init__(self) -> None:
        """Initialize loader with empty cache."""
        self._cache: SnapshotScopedCache[T] = SnapshotScopedCache()

    def load(
        self,
        gateway: StorageGateway,
        *,
        repo: str,
        commit: str,
    ) -> T:
        """Load data for a snapshot, using cache if available.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        T
            Loaded data.
        """
        key = SnapshotKey(repo=repo, commit=commit)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        data = self._load_impl(gateway, repo=repo, commit=commit)
        self._cache.set(key, data)
        return data

    @abstractmethod
    def _load_impl(
        self,
        gateway: StorageGateway,
        *,
        repo: str,
        commit: str,
    ) -> T:
        """Load data from storage.

        Subclasses must implement this method to define the actual
        loading logic.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        T
            Loaded data.
        """
        ...

    def invalidate(
        self,
        *,
        repo: str | None = None,
        commit: str | None = None,
    ) -> int:
        """Invalidate cached data matching filters.

        Parameters
        ----------
        repo
            Repository to invalidate. If None, matches any repo.
        commit
            Commit to invalidate. If None, matches any commit.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        return self._cache.invalidate(repo=repo, commit=commit)

    @property
    def cache_size(self) -> int:
        """Return number of cached entries.

        Returns
        -------
        int
            Current cache size.
        """
        return len(self._cache)


__all__ = [
    "BaseDataLoader",
]
