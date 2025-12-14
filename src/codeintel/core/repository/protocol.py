"""Repository protocol definitions.

This module defines the core protocols for repository implementations,
providing a standardized interface for data access patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.repository.pagination import PagedResult, Pagination

T = TypeVar("T")


@runtime_checkable
class RepositoryProtocol(Protocol[T]):
    """Protocol for repository implementations.

    Repositories provide a consistent interface for accessing domain
    entities from storage. This protocol defines the minimal set of
    operations that all repositories should support.

    Type Parameters
    ---------------
    T
        The type of entity this repository manages.

    Examples
    --------
    >>> class UserRepository:
    ...     def get(self, id: int | str) -> User | None:
    ...         return self._db.query(User).filter_by(id=id).first()
    ...
    ...     def list(
    ...         self,
    ...         *,
    ...         filters: Mapping[str, object] | None = None,
    ...         pagination: Pagination | None = None,
    ...     ) -> PagedResult[User]:
    ...         query = self._db.query(User)
    ...         # Apply filters and pagination
    ...         return PagedResult(items=items, total=total)
    ...
    ...     def count(self, *, filters: Mapping[str, object] | None = None) -> int:
    ...         return self._db.query(User).count()
    ...
    ...     def exists(self, id: int | str) -> bool:
    ...         return self._db.query(User).filter_by(id=id).exists()
    """

    def get(self, entity_id: int | str) -> T | None:
        """Get a single entity by ID.

        Parameters
        ----------
        entity_id
            Unique identifier for the entity.

        Returns
        -------
        T | None
            The entity if found, None otherwise.
        """
        ...

    def list(
        self,
        *,
        filters: Mapping[str, object] | None = None,
        pagination: Pagination | None = None,
    ) -> PagedResult[T]:
        """List entities with optional filtering and pagination.

        Parameters
        ----------
        filters
            Optional filters to apply.
        pagination
            Optional pagination parameters.

        Returns
        -------
        PagedResult[T]
            Paginated result with entities and metadata.
        """
        ...

    def count(self, *, filters: Mapping[str, object] | None = None) -> int:
        """Count entities matching the given filters.

        Parameters
        ----------
        filters
            Optional filters to apply.

        Returns
        -------
        int
            Number of matching entities.
        """
        ...

    def exists(self, entity_id: int | str) -> bool:
        """Check if an entity exists.

        Parameters
        ----------
        entity_id
            Unique identifier for the entity.

        Returns
        -------
        bool
            True if the entity exists.
        """
        ...


@runtime_checkable
class WriteableRepositoryProtocol(Protocol[T]):
    """Protocol for repositories that support write operations.

    Extends the basic repository interface with create, update,
    and delete operations.

    Type Parameters
    ---------------
    T
        The type of entity this repository manages.
    """

    def create(self, entity: T) -> T:
        """Create a new entity.

        Parameters
        ----------
        entity
            Entity to create.

        Returns
        -------
        T
            Created entity (may include generated fields like ID).
        """
        ...

    def update(self, entity: T) -> T:
        """Update an existing entity.

        Parameters
        ----------
        entity
            Entity with updated values.

        Returns
        -------
        T
            Updated entity.
        """
        ...

    def delete(self, entity_id: int | str) -> bool:
        """Delete an entity by ID.

        Parameters
        ----------
        entity_id
            Unique identifier for the entity.

        Returns
        -------
        bool
            True if the entity was deleted.
        """
        ...


@runtime_checkable
class BulkRepositoryProtocol(Protocol[T]):
    """Protocol for repositories that support bulk operations.

    Type Parameters
    ---------------
    T
        The type of entity this repository manages.
    """

    def bulk_create(self, entities: list[T]) -> list[T]:
        """Create multiple entities.

        Parameters
        ----------
        entities
            List of entities to create.

        Returns
        -------
        list[T]
            Created entities.
        """
        ...

    def bulk_delete(self, ids: list[int | str]) -> int:
        """Delete multiple entities by ID.

        Parameters
        ----------
        ids
            List of entity IDs to delete.

        Returns
        -------
        int
            Number of entities deleted.
        """
        ...


__all__ = [
    "BulkRepositoryProtocol",
    "RepositoryProtocol",
    "WriteableRepositoryProtocol",
]
