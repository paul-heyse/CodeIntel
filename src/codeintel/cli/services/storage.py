"""Storage gateway management service.

Consolidate gateway access from:
- ``handlers/context.py`` (_open_gateway, gateway_scope, write_gateway)
- ``deps/providers.py`` (LazyStorageProvider)
- ``handlers/_utilities.py`` (runtime_gateway)

Provide lazy initialization and proper lifecycle management.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.validation import ContractValidationMode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.cli.services.runtime import RuntimeService
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


def default_validation_summary_path(db_path: Path) -> Path | None:
    """Return the default contract validation summary path.

    Parameters
    ----------
    db_path
        Database path.

    Returns
    -------
    Path | None
        Summary path or None for in-memory databases.
    """
    if str(db_path) == ":memory:":
        return None
    name = f"{db_path.stem}.contract_validation.json"
    return db_path.with_name(name)


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
    >>> gateway = service.gateway
    >>> service.close()
    """

    SERVICE_NAME: ClassVar[str] = "storage"

    def initialize(self) -> None:
        """Initialize the service (gateway is opened lazily on first access)."""

    def shutdown(self) -> None:
        """Shut down the service by closing the gateway."""
        self.close()

    @property
    def is_ready(self) -> bool:
        """Check if service is ready.

        Returns
        -------
        bool
            True if gateway is open or service hasn't been closed.
        """
        return not self._closed

    def __init__(
        self,
        runtime: RuntimeService | None = None,
        *,
        db_path: Path | None = None,
        validation_mode: ContractValidationMode = ContractValidationMode.OFF,
    ) -> None:
        """Initialize storage service."""
        self._runtime = runtime
        self._explicit_db_path = db_path
        self._validation_mode = validation_mode
        self._gateway: StorageGateway | None = None
        self._owns_gateway = True
        self._closed = False

    @classmethod
    def from_runtime(
        cls,
        runtime: RuntimeService,
        *,
        validation_mode: ContractValidationMode = ContractValidationMode.OFF,
    ) -> StorageService:
        """Create from RuntimeService.

        Parameters
        ----------
        runtime
            Runtime service for path resolution.
        validation_mode
            Contract validation behavior for opened gateways.

        Returns
        -------
        StorageService
            Configured storage service.
        """
        return cls(runtime=runtime, validation_mode=validation_mode)

    @classmethod
    def from_path(
        cls,
        db_path: Path,
        *,
        validation_mode: ContractValidationMode = ContractValidationMode.OFF,
    ) -> StorageService:
        """Create with explicit database path.

        Parameters
        ----------
        db_path
            Database file path.
        validation_mode
            Contract validation behavior for opened gateways.

        Returns
        -------
        StorageService
            Configured storage service.
        """
        return cls(db_path=db_path, validation_mode=validation_mode)

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
        service._owns_gateway = False
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
            self._gateway = self._open_gateway(
                read_only=True,
                validation_mode=self._validation_mode,
            )
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
    def gateway_scope(
        self,
        *,
        read_only: bool = True,
        validation_mode: ContractValidationMode | None = None,
    ) -> Iterator[StorageGateway]:
        """Context manager for explicit gateway lifecycle.

        Use this when you need a gateway with a specific lifecycle that
        differs from the default lazy gateway.

        Parameters
        ----------
        read_only
            Whether to open in read-only mode.
        validation_mode
            Contract validation behavior when opening the gateway.

        Yields
        ------
        StorageGateway
            Open gateway closed on context exit.

        Examples
        --------
        >>> with service.gateway_scope(read_only=False) as gw:
        ...     gw.execute("INSERT INTO test VALUES (1)")
        """
        gateway = self._open_gateway(
            read_only=read_only,
            validation_mode=validation_mode or self._validation_mode,
        )
        if not read_only:
            gateway.policy.ensure_schemas_preserve()
        try:
            yield gateway
        finally:
            gateway.close()

    @contextmanager
    def write_gateway(
        self,
        *,
        validation_mode: ContractValidationMode | None = None,
    ) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway.

        Convenience method for write operations.

        Parameters
        ----------
        validation_mode
            Contract validation behavior when opening the gateway.

        Yields
        ------
        StorageGateway
            Write-enabled gateway closed on context exit.

        Examples
        --------
        >>> with service.write_gateway() as gw:
        ...     gw.execute("CREATE TABLE test (id INT)")
        """
        with self.gateway_scope(
            read_only=False,
            validation_mode=validation_mode,
        ) as gw:
            yield gw

    def close(self) -> None:
        """Close the cached gateway.

        Safe to call multiple times. After closing, the gateway property
        will raise RuntimeError.
        """
        if self._closed:
            return

        if self._gateway is not None and self._owns_gateway:
            try:
                self._gateway.close()
            except Exception:
                LOG.exception("Error closing gateway")
        self._gateway = None

        self._closed = True

    def _open_gateway(
        self,
        *,
        read_only: bool,
        validation_mode: ContractValidationMode,
    ) -> StorageGateway:
        """Open a new gateway.

        Parameters
        ----------
        read_only
            Whether to open in read-only mode.
        validation_mode
            Contract validation behavior when opening the gateway.

        Returns
        -------
        StorageGateway
            Open gateway.
        """
        config = self._build_config(
            read_only=read_only,
            validation_mode=validation_mode,
        )
        return open_gateway(config)

    def _build_config(
        self,
        *,
        read_only: bool,
        validation_mode: ContractValidationMode,
    ) -> StorageConfig:
        dataset_root_dir: Path | None = None
        snapshot_id: str | None = None
        repo: str | None = None
        if self._runtime is not None:
            runtime = self._runtime.runtime
            dataset_root_dir = runtime.paths.dataset_root_dir
            snapshot_id = runtime.commit
            repo = runtime.repo
        validation_summary_path = (
            None
            if validation_mode is ContractValidationMode.OFF
            else default_validation_summary_path(self.db_path)
        )
        return StorageConfig(
            db_path=self.db_path,
            dataset_root_dir=dataset_root_dir,
            read_only=read_only,
            validate_schema=validation_mode is not ContractValidationMode.OFF,
            validation_mode=validation_mode,
            validation_summary_path=validation_summary_path,
            repo=repo,
            commit=snapshot_id,
        )

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
    "default_validation_summary_path",
]
