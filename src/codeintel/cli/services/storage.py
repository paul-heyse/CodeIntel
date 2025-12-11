"""Storage gateway management service.

Consolidate gateway access from:
- ``handlers/context.py`` (_open_gateway, gateway_scope, write_gateway)
- ``deps/providers.py`` (LazyStorageProvider)
- ``handlers/_utilities.py`` (runtime_gateway)

Provide lazy initialization and proper lifecycle management.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Self

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.services.runtime import RuntimeService

LOG = logging.getLogger(__name__)


class StorageService:
    """Storage gateway lifecycle management.

    Provide lazy-loaded gateway access with automatic lifecycle management.
    The gateway is opened on first access and closed when the service is closed.

    Parameters
    ----------
    runtime
        RuntimeService for database path resolution.
    db_path
        Optional explicit database path (overrides runtime).

    Examples
    --------
    >>> service = StorageService.from_path(Path("build/db/codeintel.duckdb"))
    >>> gateway = service.gateway  # Lazy open
    >>> service.close()  # Cleanup
    """

    def __init__(
        self,
        runtime: RuntimeService | None = None,
        *,
        db_path: Path | None = None,
    ) -> None:
        """Initialize storage service."""
        self._runtime = runtime
        self._explicit_db_path = db_path
        self._gateway: StorageGateway | None = None
        self._closed = False

    @classmethod
    def from_runtime(cls, runtime: RuntimeService) -> StorageService:
        """Create from RuntimeService.

        Parameters
        ----------
        runtime
            Runtime service for path resolution.

        Returns
        -------
        StorageService
            Configured storage service.
        """
        return cls(runtime=runtime)

    @classmethod
    def from_path(cls, db_path: Path) -> StorageService:
        """Create with explicit database path.

        Parameters
        ----------
        db_path
            Database file path.

        Returns
        -------
        StorageService
            Configured storage service.
        """
        return cls(db_path=db_path)

    @classmethod
    def from_gateway(cls, gateway: StorageGateway) -> StorageService:
        """Create with a pre-opened gateway (for testing).

        Parameters
        ----------
        gateway
            Pre-opened storage gateway.

        Returns
        -------
        StorageService
            Storage service wrapping the gateway.
        """
        service = cls()
        service._gateway = gateway
        return service

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (lazy, read-only).

        The gateway is opened on first access and cached.

        Returns
        -------
        StorageGateway
            Open storage gateway.

        Raises
        ------
        RuntimeError
            If service has been closed.
        """
        if self._closed:
            msg = "StorageService has been closed"
            raise RuntimeError(msg)

        if self._gateway is None:
            self._gateway = self._open_gateway(read_only=True)
        return self._gateway

    @property
    def db_path(self) -> Path:
        """Get database path.

        Returns
        -------
        Path
            Database file path.

        Raises
        ------
        RuntimeError
            If no database path is available.
        """
        if self._explicit_db_path is not None:
            return self._explicit_db_path
        if self._runtime is not None:
            return self._runtime.db_path
        msg = "No database path available - provide runtime or explicit path"
        raise RuntimeError(msg)

    @property
    def is_open(self) -> bool:
        """Check if gateway is currently open.

        Returns
        -------
        bool
            True if gateway is open.
        """
        return self._gateway is not None and not self._closed

    @contextmanager
    def gateway_scope(self, *, read_only: bool = True) -> Iterator[StorageGateway]:
        """Context manager for explicit gateway lifecycle.

        Use this when you need a gateway with a specific lifecycle that
        differs from the default lazy gateway.

        Parameters
        ----------
        read_only
            Whether to open in read-only mode.

        Yields
        ------
        StorageGateway
            Open gateway closed on context exit.

        Examples
        --------
        >>> with service.gateway_scope(read_only=False) as gw:  # doctest: +SKIP
        ...     gw.execute("INSERT INTO test VALUES (1)")
        """
        gateway = self._open_gateway(read_only=read_only)
        try:
            yield gateway
        finally:
            gateway.close()

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway.

        Convenience method for write operations.

        Yields
        ------
        StorageGateway
            Write-enabled gateway closed on context exit.

        Examples
        --------
        >>> with service.write_gateway() as gw:  # doctest: +SKIP
        ...     gw.execute("CREATE TABLE test (id INT)")
        """
        with self.gateway_scope(read_only=False) as gw:
            yield gw

    def close(self) -> None:
        """Close the cached gateway.

        Safe to call multiple times. After closing, the gateway property
        will raise RuntimeError.
        """
        if self._closed:
            return

        if self._gateway is not None:
            try:
                self._gateway.close()
            except Exception:
                LOG.exception("Error closing gateway")
            self._gateway = None

        self._closed = True

    def _open_gateway(self, *, read_only: bool) -> StorageGateway:
        """Open a new gateway.

        Parameters
        ----------
        read_only
            Whether to open in read-only mode.

        Returns
        -------
        StorageGateway
            Open gateway.
        """
        config = StorageConfig(db_path=self.db_path, read_only=read_only)
        return open_gateway(config)

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns
        -------
        Self
            Self for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing gateway."""
        self.close()


__all__ = [
    "StorageService",
]
