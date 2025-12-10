"""Dependency container for command execution.

Provide the Deps dataclass that commands receive as their execution context,
and the DepsBuilder for constructing it with appropriate providers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.deps.protocols import JobManagerProtocol, ServingAccess, StorageAccess
from codeintel.cli.deps.providers import LazyServingProvider, LazyStorageProvider

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig


@dataclass
class Deps:
    """Container for command dependencies.

    Commands receive this as their execution context. Dependencies are
    lazy-loaded based on what the command actually accesses.

    Parameters
    ----------
    config
        CLI configuration.
    logger
        Logger for command output.
    jobs
        Job manager for background job operations.

    Examples
    --------
    >>> def execute(self, deps: Deps) -> CliResult[T]:
    ...     # Storage only loaded if accessed
    ...     rows = deps.storage.gateway.query("SELECT * FROM t")
    ...
    ...     # Jobs always available (lightweight)
    ...     jobs = deps.jobs.list_jobs(limit=10)
    """

    config: CliConfig
    logger: logging.Logger
    jobs: JobManagerProtocol

    # Optional/lazy-loaded dependencies
    _storage: StorageAccess | None = field(default=None, repr=False)
    _serving: ServingAccess | None = field(default=None, repr=False)

    @property
    def storage(self) -> StorageAccess:
        """Get storage access.

        Returns
        -------
        StorageAccess
            Storage access provider.

        Raises
        ------
        RuntimeError
            If storage was not configured for this command.
        """
        if self._storage is None:
            msg = "Storage not available. Command must declare require_storage=True"
            raise RuntimeError(msg)
        return self._storage

    @property
    def serving(self) -> ServingAccess:
        """Get serving access.

        Returns
        -------
        ServingAccess
            Serving access provider.

        Raises
        ------
        RuntimeError
            If serving was not configured for this command.
        """
        if self._serving is None:
            msg = "Serving not available. Command must declare require_serving=True"
            raise RuntimeError(msg)
        return self._serving

    @property
    def has_storage(self) -> bool:
        """Check if storage access is available.

        Returns
        -------
        bool
            True if storage was configured.
        """
        return self._storage is not None

    @property
    def has_serving(self) -> bool:
        """Check if serving access is available.

        Returns
        -------
        bool
            True if serving was configured.
        """
        return self._serving is not None


class DepsBuilder:
    """Builder for constructing Deps with appropriate providers.

    Used by @cli_command to build dependencies based on command requirements.

    Examples
    --------
    >>> with DepsBuilder().with_storage().build() as deps:
    ...     result = command.execute(deps)
    """

    def __init__(self) -> None:
        """Initialize deps builder with default settings."""
        self._require_storage = False
        self._require_serving = False
        self._project_root: Path | None = None
        self._db_path: Path | None = None
        self._verbosity: int = 0

    def with_storage(self, *, db_path: Path | None = None) -> DepsBuilder:
        """Enable storage access.

        Parameters
        ----------
        db_path
            Optional explicit database path.

        Returns
        -------
        DepsBuilder
            Self for chaining.
        """
        self._require_storage = True
        if db_path is not None:
            self._db_path = db_path
        return self

    def with_serving(self) -> DepsBuilder:
        """Enable serving access.

        Returns
        -------
        DepsBuilder
            Self for chaining.
        """
        self._require_serving = True
        return self

    def with_project(self, root: Path | None) -> DepsBuilder:
        """Set project root for resolution.

        Parameters
        ----------
        root
            Project root directory.

        Returns
        -------
        DepsBuilder
            Self for chaining.
        """
        self._project_root = root
        return self

    def with_verbosity(self, level: int) -> DepsBuilder:
        """Set verbosity level for logging.

        Parameters
        ----------
        level
            Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).

        Returns
        -------
        DepsBuilder
            Self for chaining.
        """
        self._verbosity = level
        return self

    @contextmanager
    def build(self) -> Iterator[Deps]:
        """Build Deps and manage resource lifecycle.

        Yields
        ------
        Deps
            Configured dependency container.
        """
        from codeintel.cli.config import load_config
        from codeintel.cli.jobs import get_job_manager

        config = load_config(validate=False)
        logger = logging.getLogger("codeintel.cli")
        jobs = get_job_manager()

        storage: LazyStorageProvider | None = None
        serving: LazyServingProvider | None = None

        try:
            if self._require_storage:
                storage = LazyStorageProvider(
                    project_root=self._project_root,
                    db_path=self._db_path,
                )

            if self._require_serving:
                serving = LazyServingProvider(
                    storage=storage,
                    project_root=self._project_root,
                )

            yield Deps(
                config=config,
                logger=logger,
                jobs=jobs,
                _storage=storage,
                _serving=serving,
            )
        finally:
            # Cleanup resources
            if storage is not None:
                storage.close()


__all__ = [
    "Deps",
    "DepsBuilder",
]
