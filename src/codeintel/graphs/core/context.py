"""Graph plugin execution context.

This module defines the execution context provided to graph plugins,
providing access to storage, configuration, and shared scratch space
without any dependency on the analytics subsystem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.engine import GraphEngine
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway


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

    Provides access to storage, graph engine, and shared scratch space
    without any dependency on the analytics subsystem.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    snapshot
        Repository snapshot reference.
    engine
        GraphEngine for accessing cached graphs.
    paths
        Build paths configuration.
    catalog_provider
        Optional function catalog provider.
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
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    engine: GraphEngine | None = None
    paths: BuildPaths | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    scratch: GraphRuntimeScratch = field(default_factory=GraphRuntimeScratch)
    options: object | None = None
    plugin_name: str | None = None
    run_id: str | None = None
    scope: GraphRunScope | None = None

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


__all__ = [
    "GraphExecutionContext",
    "GraphRuntimeScratch",
]
