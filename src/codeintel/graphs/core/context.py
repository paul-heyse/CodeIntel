"""Graph plugin execution context.

This module defines the execution context provided to graph plugins,
providing access to storage, configuration, and shared scratch space
without any dependency on the analytics subsystem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.graphs.engine import GraphEngine
    from codeintel.graphs.resources.container import ResourceContainer
    from codeintel.graphs.resources.protocol import ResourceProvider
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@dataclass
class GraphRuntimeScratch:
    """Ephemeral scratch/cache store shared across plugin executions in a run.

    Provides a way for plugins to share intermediate data within a single
    execution run without persisting to the database.
    """

    _store: dict[str, object] = field(default_factory=dict)
    _cleanup: list[Callable[[], None]] = field(default_factory=list)

    def declare(self, key: str, value: object) -> None:
        """Record a value for later consumption by other plugins.

        Parameters
        ----------
        key
            Identifier for the stored value.
        value
            Value to store.
        """
        self._store[key] = value

    def consume(self, key: str, default: object | None = None) -> object | None:
        """Retrieve a value populated by another plugin.

        Parameters
        ----------
        key
            Identifier of the value to retrieve.
        default
            Value to return if key is not found.

        Returns
        -------
        object | None
            Cached value or provided default.
        """
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the scratch store.

        Parameters
        ----------
        key
            Identifier to check.

        Returns
        -------
        bool
            True if key exists.
        """
        return key in self._store

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback executed after the run completes.

        Parameters
        ----------
        callback
            Function to call during cleanup.
        """
        self._cleanup.append(callback)

    def cleanup(self) -> None:
        """Execute cleanup callbacks and clear stored values."""
        import logging  # noqa: PLC0415

        log = logging.getLogger(__name__)
        for callback in reversed(self._cleanup):
            try:
                callback()
            except (RuntimeError, OSError, ValueError):
                log.exception("scratch.cleanup_failed")
        self._store.clear()
        self._cleanup.clear()

    def __len__(self) -> int:
        """Return the number of declared cache entries.

        Returns
        -------
        int
            Count of cached entries.
        """
        return len(self._store)

    def keys(self) -> tuple[str, ...]:
        """Return declared cache keys.

        Returns
        -------
        tuple[str, ...]
            Cache key names.
        """
        return tuple(self._store.keys())


@dataclass
class GraphExecutionContext:
    """Execution context for graph plugins.

    Provides access to storage, graph engine, resource providers, and
    shared scratch space without any dependency on the analytics subsystem.

    All I/O access should go through the resource container via `require()`.
    The private attributes `_gateway`, `_engine`, and `_catalog_provider` are
    only for internal initialization and should not be accessed directly.

    Attributes
    ----------
    snapshot
        Repository snapshot reference.
    resources
        Resource container for dependency injection (required).
    paths
        Build paths configuration.
    scratch
        Shared scratch space for inter-plugin data.
    options
        Plugin-specific options.
    plugin_name
        Name of the executing plugin.
    run_id
        Unique identifier for this execution run.
    scope
        Optional scoping for incremental execution.
    run_context
        Optional unified run context for cross-engine correlation.
    """

    snapshot: SnapshotRef
    resources: ResourceContainer
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _engine: GraphEngine | None = field(default=None, repr=False)
    _catalog_provider: FunctionCatalogProvider | None = field(default=None, repr=False)
    paths: BuildPaths | None = None
    scratch: GraphRuntimeScratch = field(default_factory=GraphRuntimeScratch)
    options: object | None = None
    plugin_name: str | None = None
    run_id: str | None = None
    scope: GraphRunScope | None = None
    run_context: RunContext | None = None

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot.

        Returns
        -------
        str
            Commit hash or identifier.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        return self.snapshot.repo_root

    @property
    def effective_run_id(self) -> str | None:
        """Get run ID preferring unified RunContext if present.

        Returns
        -------
        str | None
            Run ID from run_context if set, otherwise falls back to run_id.
        """
        if self.run_context is not None:
            return self.run_context.run_id
        return self.run_id

    @property
    def gateway(self) -> StorageGateway:
        """Get the storage gateway.

        Tries resource injection first, falls back to private attribute.

        Returns
        -------
        StorageGateway
            Storage gateway for database access.

        Raises
        ------
        RuntimeError
            If no gateway is available.
        """
        # Try resource injection first
        from codeintel.graphs.resources.storage import StorageResource  # noqa: PLC0415

        if self.resources.has(StorageResource.RESOURCE_NAME):
            storage = self.resources.require(StorageResource)
            return storage.gateway
        # Fall back to private attribute
        if self._gateway is not None:
            return self._gateway
        msg = "No gateway available in context"
        raise RuntimeError(msg)

    @property
    def engine(self) -> GraphEngine | None:
        """Get the graph engine.

        Tries resource injection first, falls back to private attribute.

        Returns
        -------
        GraphEngine | None
            Graph engine if available.
        """
        # Try resource injection first
        from codeintel.graphs.resources.graphs import GraphResource  # noqa: PLC0415

        if self.resources.has(GraphResource.RESOURCE_NAME):
            graph_resource = self.resources.require(GraphResource)
            return graph_resource.engine
        # Fall back to private attribute
        return self._engine

    @property
    def catalog_provider(self) -> FunctionCatalogProvider | None:
        """Get the function catalog provider.

        Returns
        -------
        FunctionCatalogProvider | None
            Catalog provider if available.
        """
        return self._catalog_provider

    def require(self, provider_type: type[ResourceProvider[T]]) -> T:
        """Get a resource from the container.

        Parameters
        ----------
        provider_type
            The resource provider type to look up.

        Returns
        -------
        T
            The resource value.

        Notes
        -----
        May raise ``ResourceNotFoundError`` if the resource is not registered.
        """
        return self.resources.require(provider_type)

    def require_by_name(self, name: str) -> object:
        """Get a resource by name from the container.

        Parameters
        ----------
        name
            Resource name to look up.

        Returns
        -------
        object
            The resource value.

        Notes
        -----
        May raise ``ResourceNotFoundError`` if the resource is not registered.
        """
        return self.resources.require_by_name(name)

    def has_resource(self, name: str) -> bool:
        """Check if a resource is available.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return self.resources.has(name)


__all__ = [
    "GraphExecutionContext",
    "GraphRuntimeScratch",
]
