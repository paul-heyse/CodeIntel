"""Protocol interfaces for execution contexts.

This module defines protocol interfaces that can be implemented by
different execution context types across analytics, graphs, and ingestion.

These protocols enable type-safe programming against context interfaces
without coupling to specific implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@runtime_checkable
class SnapshotContextProtocol(Protocol):
    """Protocol for contexts providing snapshot information.

    Implementations provide access to repository and commit identifiers
    along with the repository root path.
    """

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference.

        Returns
        -------
        SnapshotRef
            Repository snapshot reference.
        """
        ...

    @property
    def repo(self) -> str:
        """Return repository identifier.

        Returns
        -------
        str
            Repository slug.
        """
        ...

    @property
    def commit(self) -> str:
        """Return commit identifier.

        Returns
        -------
        str
            Commit hash.
        """
        ...

    @property
    def repo_root(self) -> Path:
        """Return repository root path.

        Returns
        -------
        Path
            Absolute path to repository root.
        """
        ...


@runtime_checkable
class StorageContextProtocol(Protocol):
    """Protocol for contexts providing storage gateway access.

    Implementations provide access to the storage gateway for
    database operations.
    """

    @property
    def gateway(self) -> StorageGateway:
        """Return storage gateway.

        Returns
        -------
        StorageGateway
            Storage gateway for database access.
        """
        ...


@runtime_checkable
class ConfigContextProtocol(Protocol):
    """Protocol for contexts providing configuration access.

    Implementations provide typed access to configuration objects.
    """

    def get_config(self, config_type: type[T]) -> T:
        """Return configuration of the requested type.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T
            Configuration instance.

        Raises
        ------
        ValueError
            If the configuration is not available.
        """
        ...

    def get_optional_config(self, config_type: type[T]) -> T | None:
        """Return configuration if available.

        Parameters
        ----------
        config_type
            Type of configuration to retrieve.

        Returns
        -------
        T | None
            Configuration instance or None.
        """
        ...

    def has_config(self, config_type: type[T]) -> bool:
        """Check if configuration is available.

        Parameters
        ----------
        config_type
            Type to check.

        Returns
        -------
        bool
            True if configuration is available.
        """
        ...


@runtime_checkable
class ResourceContextProtocol(Protocol):
    """Protocol for contexts providing resource registry access.

    Implementations provide typed access to resources through
    a resource registry.
    """

    def require(self, resource_type: type[T]) -> T:
        """Get a required resource.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        ResourceNotFoundError
            If the resource is not available.
        """
        ...

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get a resource or None if unavailable.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T | None
            The resource, or None if unavailable.
        """
        ...

    def has_resource(self, resource_type: type) -> bool:
        """Check if a resource type is available.

        Parameters
        ----------
        resource_type
            Type to check.

        Returns
        -------
        bool
            True if the resource is available.
        """
        ...


@runtime_checkable
class ExecutionContextProtocol(
    SnapshotContextProtocol,
    StorageContextProtocol,
    ConfigContextProtocol,
    ResourceContextProtocol,
    Protocol,
):
    """Complete protocol combining all context capabilities.

    This protocol represents a full execution context with access
    to snapshot information, storage gateway, configuration, and
    resources.

    Implementations include:
    - PluginExecutionContext (core plugins)
    - GraphContext (analytics runtime)
    - IngestContext (ingestion runtime)
    """

    @property
    def run_id(self) -> str | None:
        """Return run identifier.

        Returns
        -------
        str | None
            Unique run identifier, or None if not set.
        """
        ...


__all__ = [
    "ConfigContextProtocol",
    "ExecutionContextProtocol",
    "ResourceContextProtocol",
    "SnapshotContextProtocol",
    "StorageContextProtocol",
]
