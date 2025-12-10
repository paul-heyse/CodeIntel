"""Lazy resource providers for operations.

Providers defer resource initialization until first access,
ensuring lightweight context creation for operations that don't
need certain resources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.jobs._jobs import JobManager
    from codeintel.storage.gateway import StorageGateway


LOG = logging.getLogger(__name__)


@dataclass
class LazyStorageProvider:
    """Lazy provider for StorageGateway.

    Defers gateway creation until first access, then caches.

    Parameters
    ----------
    db_path
        Path to the database.
    read_only
        Whether to open in read-only mode.
    _gateway
        Cached gateway instance.

    Example
    -------
    >>> provider = LazyStorageProvider(Path("/path/to/db.duckdb"))
    >>> gateway = provider.invoke()  # Creates gateway
    >>> gateway2 = provider.invoke()  # Returns cached
    >>> gateway is gateway2
    True
    """

    db_path: Path | None = None
    read_only: bool = False
    _gateway: StorageGateway | None = field(default=None, repr=False)

    def invoke(self) -> StorageGateway:
        """Get or create the storage gateway.

        Returns
        -------
        StorageGateway
            The storage gateway.

        Raises
        ------
        RuntimeError
            If db_path is not configured.
        """
        if self._gateway is not None:
            return self._gateway

        if self.db_path is None:
            msg = "db_path is required for storage access"
            raise RuntimeError(msg)

        # Import here to avoid circular imports (intentional deferred import)
        from codeintel.storage.gateway import open_gateway  # noqa: PLC0415

        LOG.debug("Opening storage gateway at %s (read_only=%s)", self.db_path, self.read_only)
        self._gateway = open_gateway(self.db_path, read_only=self.read_only)
        return self._gateway

    def close(self) -> None:
        """Close the gateway if open."""
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None


@dataclass
class LazyJobsProvider:
    """Lazy provider for JobManager.

    Parameters
    ----------
    _manager
        Cached job manager instance.

    Example
    -------
    >>> provider = LazyJobsProvider()
    >>> manager = provider.invoke()
    """

    _manager: JobManager | None = field(default=None, repr=False)

    def invoke(self) -> JobManager:
        """Get or create the job manager.

        Returns
        -------
        JobManager
            The job manager.
        """
        if self._manager is not None:
            return self._manager

        # Import here to avoid circular imports (intentional deferred import)
        from codeintel.cli.jobs._jobs import get_job_manager  # noqa: PLC0415

        LOG.debug("Creating job manager")
        self._manager = get_job_manager()
        return self._manager


@dataclass
class TelemetryContextImpl:
    """Simple telemetry context implementation.

    Logs telemetry events when OpenTelemetry is not available.

    Parameters
    ----------
    _logger
        Logger for telemetry events.
    """

    _logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("codeintel.telemetry")
    )

    def start_span(self, name: str) -> object:
        """Start a new trace span.

        Parameters
        ----------
        name
            Span name.

        Returns
        -------
        object
            Span context (None for this implementation).
        """
        self._logger.debug("span.start: %s", name)
        return None

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
        """Add an event to the current span.

        Parameters
        ----------
        name
            Event name.
        attributes
            Optional event attributes.
        """
        self._logger.debug("span.event: %s %s", name, attributes or {})

    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric value.

        Parameters
        ----------
        name
            Metric name.
        value
            Metric value.
        tags
            Optional tags.
        """
        self._logger.debug("metric: %s=%s %s", name, value, tags or {})


__all__ = [
    "LazyJobsProvider",
    "LazyStorageProvider",
    "TelemetryContextImpl",
]
