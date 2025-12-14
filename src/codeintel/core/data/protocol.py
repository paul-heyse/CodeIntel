"""Data loader protocol definitions.

This module provides the DataLoaderProtocol for types that load
snapshot-scoped data from storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class DataLoaderProtocol[T](Protocol):
    """Protocol for snapshot-scoped data loaders.

    Implementations load data for a specific repository snapshot
    (repo + commit) with optional caching.

    Examples
    --------
    >>> class FunctionLoader:
    ...     def load(
    ...         self,
    ...         gateway: StorageGateway,
    ...         *,
    ...         repo: str,
    ...         commit: str,
    ...     ) -> list[Function]:
    ...         # Load function data from database
    ...         return []
    ...
    ...     def invalidate(
    ...         self,
    ...         *,
    ...         repo: str | None = None,
    ...         commit: str | None = None,
    ...     ) -> None:
    ...         # Clear cached data
    ...         pass
    """

    def load(
        self,
        gateway: StorageGateway,
        *,
        repo: str,
        commit: str,
    ) -> T:
        """Load data for a snapshot.

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
    ) -> None:
        """Invalidate cached data.

        Parameters
        ----------
        repo
            Optional repository to invalidate. If None, invalidates all repos.
        commit
            Optional commit to invalidate. If None, invalidates all commits.
        """
        ...


__all__ = [
    "DataLoaderProtocol",
]
