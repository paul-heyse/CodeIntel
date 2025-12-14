"""Scoped cache implementations.

This module provides cache implementations that support scoped
invalidation, such as by repository or snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")


@dataclass(frozen=True)
class SnapshotKey:
    """Key for snapshot-scoped data.

    Represents a unique repository snapshot identified by
    repository name and commit hash.

    Attributes
    ----------
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit hash or identifier.

    Examples
    --------
    >>> key = SnapshotKey(repo="myorg/myrepo", commit="abc123")
    >>> key.as_tuple()
    ('myorg/myrepo', 'abc123')
    """

    repo: str
    commit: str

    def as_tuple(self) -> tuple[str, str]:
        """Return as (repo, commit) tuple.

        Returns
        -------
        tuple[str, str]
            Tuple of (repo, commit).
        """
        return (self.repo, self.commit)

    def matches(
        self,
        *,
        repo: str | None = None,
        commit: str | None = None,
    ) -> bool:
        """Check if key matches the given filters.

        Parameters
        ----------
        repo
            Repository to match. If None, matches any repo.
        commit
            Commit to match. If None, matches any commit.

        Returns
        -------
        bool
            True if key matches all provided filters.
        """
        repo_matches = repo is None or self.repo == repo
        commit_matches = commit is None or self.commit == commit
        return repo_matches and commit_matches


class SnapshotScopedCache[T]:
    """Cache for snapshot-scoped data.

    Provides a simple cache keyed by (repo, commit) pairs with
    support for selective invalidation.

    Examples
    --------
    >>> cache: SnapshotScopedCache[list[str]] = SnapshotScopedCache()
    >>> key = SnapshotKey(repo="org/repo", commit="abc")
    >>> cache.set(key, ["item1", "item2"])
    >>> cache.get(key)
    ['item1', 'item2']
    >>> cache.invalidate(repo="org/repo")
    1
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[SnapshotKey, T] = {}

    def get(self, key: SnapshotKey) -> T | None:
        """Get cached value for key.

        Parameters
        ----------
        key
            Snapshot key.

        Returns
        -------
        T | None
            Cached value, or None if not present.
        """
        return self._cache.get(key)

    def set(self, key: SnapshotKey, value: T) -> None:
        """Set cached value for key.

        Parameters
        ----------
        key
            Snapshot key.
        value
            Value to cache.
        """
        self._cache[key] = value

    def has(self, key: SnapshotKey) -> bool:
        """Check if key is in cache.

        Parameters
        ----------
        key
            Snapshot key.

        Returns
        -------
        bool
            True if key is cached.
        """
        return key in self._cache

    def invalidate(
        self,
        *,
        repo: str | None = None,
        commit: str | None = None,
    ) -> int:
        """Invalidate cached entries matching filters.

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
        if repo is None and commit is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove = [k for k in self._cache if k.matches(repo=repo, commit=commit)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> int:
        """Clear all cached entries.

        Returns
        -------
        int
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def keys(self) -> Iterator[SnapshotKey]:
        """Iterate over cached keys.

        Yields
        ------
        SnapshotKey
            Each cached key.
        """
        yield from self._cache.keys()

    def __len__(self) -> int:
        """Return number of cached entries.

        Returns
        -------
        int
            Cache size.
        """
        return len(self._cache)


class TypedScopedCache[K, T]:
    """Generic scoped cache with typed keys.

    A more flexible scoped cache that works with any key type
    that supports matching via a `matches` method.

    Type Parameters
    ---------------
    K
        Key type (must have a `matches` method).
    T
        Value type.

    Examples
    --------
    >>> cache: TypedScopedCache[SnapshotKey, str] = TypedScopedCache()
    >>> cache.set(SnapshotKey("repo", "abc"), "data")
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[K, T] = {}

    def get(self, key: K) -> T | None:
        """Get cached value for key.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        T | None
            Cached value, or None if not present.
        """
        return self._cache.get(key)

    def set(self, key: K, value: T) -> None:
        """Set cached value for key.

        Parameters
        ----------
        key
            Cache key.
        value
            Value to cache.
        """
        self._cache[key] = value

    def has(self, key: K) -> bool:
        """Check if key is in cache.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        bool
            True if key is cached.
        """
        return key in self._cache

    def invalidate_matching(self, **criteria: object) -> int:
        """Invalidate entries matching the given criteria.

        Parameters
        ----------
        **criteria
            Key-value pairs to match against cached keys.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        if not criteria:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove: list[K] = []
        for key in self._cache:
            match_func = getattr(key, "matches", None)
            if callable(match_func) and match_func(**criteria):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> int:
        """Clear all cached entries.

        Returns
        -------
        int
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def __len__(self) -> int:
        """Return number of cached entries.

        Returns
        -------
        int
            Cache size.
        """
        return len(self._cache)


__all__ = [
    "SnapshotKey",
    "SnapshotScopedCache",
    "TypedScopedCache",
]
