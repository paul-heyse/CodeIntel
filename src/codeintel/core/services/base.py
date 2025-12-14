"""Base service implementation.

This module provides a base class for services with common patterns
for lifecycle management, caching, and error handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

from codeintel.core.services.protocol import HealthStatus, ServiceState

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base exception for service-related errors.

    Attributes
    ----------
    service_name
        Name of the service that raised the error.
    """

    def __init__(self, service_name: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        service_name
            Name of the service.
        message
            Error message.
        """
        super().__init__(f"[{service_name}] {message}")
        self.service_name = service_name


class ServiceInitializationError(ServiceError):
    """Raised when service initialization fails."""


class ServiceNotReadyError(ServiceError):
    """Raised when attempting to use a service that is not ready."""


class BaseService:
    """Base class for service implementations.

    Provides common patterns for service lifecycle management including
    state tracking, lazy initialization, and proper shutdown handling.

    Subclasses should override `_do_initialize()` and `_do_shutdown()`
    to implement their specific logic.

    Attributes
    ----------
    SERVICE_NAME
        Unique identifier for the service type. Must be overridden.

    Examples
    --------
    >>> class DatabaseService(BaseService):
    ...     SERVICE_NAME: ClassVar[str] = "database"
    ...
    ...     def _do_initialize(self) -> None:
    ...         self._connection = connect_to_db()
    ...
    ...     def _do_shutdown(self) -> None:
    ...         self._connection.close()
    """

    SERVICE_NAME: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize the base service."""
        self._state = ServiceState.CREATED
        self._error: Exception | None = None

    @property
    def state(self) -> ServiceState:
        """Return the current service state.

        Returns
        -------
        ServiceState
            Current state of the service.
        """
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check whether the service is ready.

        Returns
        -------
        bool
            True if the service is in READY state.
        """
        return self._state == ServiceState.READY

    @property
    def is_stopped(self) -> bool:
        """Check whether the service has been stopped.

        Returns
        -------
        bool
            True if the service is in STOPPED state.
        """
        return self._state == ServiceState.STOPPED

    @property
    def is_failed(self) -> bool:
        """Check whether the service has failed.

        Returns
        -------
        bool
            True if the service is in FAILED state.
        """
        return self._state == ServiceState.FAILED

    @property
    def last_error(self) -> Exception | None:
        """Return the last error that occurred.

        Returns
        -------
        Exception | None
            The last error, or None if no error occurred.
        """
        return self._error

    def initialize(self) -> None:
        """Initialize the service.

        Transitions the service from CREATED to READY state.
        If initialization fails, transitions to FAILED state.

        Raises
        ------
        ServiceInitializationError
            If initialization fails.
        """
        if self._state == ServiceState.READY:
            log.debug("%s already initialized", self.SERVICE_NAME)
            return

        if self._state not in {ServiceState.CREATED, ServiceState.FAILED}:
            msg = f"Cannot initialize from state {self._state.value}"
            raise ServiceInitializationError(self.SERVICE_NAME, msg)

        self._state = ServiceState.INITIALIZING
        log.debug("Initializing %s", self.SERVICE_NAME)

        try:
            self._do_initialize()
        except Exception as e:
            self._state = ServiceState.FAILED
            self._error = e
            log.exception("Failed to initialize %s", self.SERVICE_NAME)
            raise ServiceInitializationError(self.SERVICE_NAME, str(e)) from e
        else:
            self._state = ServiceState.READY
            log.info("%s initialized successfully", self.SERVICE_NAME)

    def shutdown(self) -> None:
        """Shut down the service.

        Transitions the service to STOPPED state. Safe to call multiple times.
        """
        if self._state == ServiceState.STOPPED:
            return

        if self._state == ServiceState.SHUTTING_DOWN:
            return

        self._state = ServiceState.SHUTTING_DOWN
        log.debug("Shutting down %s", self.SERVICE_NAME)

        try:
            self._do_shutdown()
        except Exception:
            log.exception("Error during %s shutdown", self.SERVICE_NAME)
        finally:
            self._state = ServiceState.STOPPED
            log.info("%s shut down", self.SERVICE_NAME)

    def ensure_ready(self) -> None:
        """Ensure the service is ready, initializing if needed.

        Raises
        ------
        ServiceNotReadyError
            If the service cannot be made ready.
        """
        if self._state == ServiceState.READY:
            return

        if self._state == ServiceState.STOPPED:
            msg = "Service has been stopped and cannot be restarted"
            raise ServiceNotReadyError(self.SERVICE_NAME, msg)

        if self._state == ServiceState.CREATED:
            self.initialize()
            return

        if self._state == ServiceState.FAILED:
            msg = f"Service failed: {self._error}"
            raise ServiceNotReadyError(self.SERVICE_NAME, msg)

        msg = f"Service is in unexpected state: {self._state.value}"
        raise ServiceNotReadyError(self.SERVICE_NAME, msg)

    def health_check(self) -> HealthStatus:
        """Perform a health check on the service.

        Override this method to provide detailed health information.

        Returns
        -------
        HealthStatus
            Health status of the service.
        """
        if self._state == ServiceState.READY:
            return HealthStatus(healthy=True, message="Service is ready")

        if self._state == ServiceState.FAILED:
            return HealthStatus(
                healthy=False,
                message=f"Service failed: {self._error}",
                details={"state": self._state.value},
            )

        return HealthStatus(
            healthy=False,
            message=f"Service not ready: {self._state.value}",
            details={"state": self._state.value},
        )

    def _do_initialize(self) -> None:
        """Perform service-specific initialization.

        Subclasses should override this method to implement
        their initialization logic.
        """

    def _do_shutdown(self) -> None:
        """Perform service-specific shutdown.

        Subclasses should override this method to implement
        their cleanup logic.
        """

    def __enter__(self) -> Self:
        """Enter context manager, initializing service.

        Returns
        -------
        Self
            The initialized service.
        """
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, shutting down service."""
        self.shutdown()


class LazyService(BaseService):
    """Service that initializes lazily on first use.

    This service variant defers initialization until the service
    is actually needed, reducing startup time for unused services.

    Examples
    --------
    >>> class ExpensiveService(LazyService):
    ...     SERVICE_NAME = "expensive"
    ...
    ...     def _do_initialize(self) -> None:
    ...         # Expensive setup only when needed
    ...         self._resource = load_expensive_resource()
    ...
    ...     def get_resource(self) -> object:
    ...         self.ensure_ready()  # Lazy init
    ...         return self._resource
    """

    def __init__(self, *, auto_initialize: bool = True) -> None:
        """Initialize lazy service.

        Parameters
        ----------
        auto_initialize
            If True, automatically initialize on first ensure_ready().
        """
        super().__init__()
        self._auto_initialize = auto_initialize

    def ensure_ready(self) -> None:
        """Ensure service is ready, auto-initializing if configured.

        May raise exceptions from initialize() or parent ensure_ready().
        """
        if self._state == ServiceState.READY:
            return

        if self._auto_initialize and self._state == ServiceState.CREATED:
            self.initialize()
            return

        super().ensure_ready()


class CachedService(BaseService):
    """Service with built-in caching support.

    Provides a simple caching mechanism for service results
    with optional invalidation callback.
    """

    def __init__(self) -> None:
        """Initialize cached service."""
        super().__init__()
        self._cache: dict[str, object] = {}
        self._invalidation_callbacks: list[Callable[[], None]] = []

    def cache_get(self, key: str) -> object | None:
        """Get a cached value.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        object | None
            Cached value or None if not found.
        """
        return self._cache.get(key)

    def cache_set(self, key: str, value: object) -> None:
        """Set a cached value.

        Parameters
        ----------
        key
            Cache key.
        value
            Value to cache.
        """
        self._cache[key] = value

    def cache_invalidate(self, key: str | None = None) -> None:
        """Invalidate cached values.

        Parameters
        ----------
        key
            Specific key to invalidate, or None to clear all.
        """
        if key is None:
            self._cache.clear()
            for callback in self._invalidation_callbacks:
                try:
                    callback()
                except Exception:
                    log.exception("Error in invalidation callback")
        else:
            self._cache.pop(key, None)

    def on_invalidate(self, callback: Callable[[], None]) -> None:
        """Register an invalidation callback.

        Parameters
        ----------
        callback
            Function to call when cache is invalidated.
        """
        self._invalidation_callbacks.append(callback)

    def _do_shutdown(self) -> None:
        """Clear cache on shutdown."""
        self._cache.clear()
        self._invalidation_callbacks.clear()


__all__ = [
    "BaseService",
    "CachedService",
    "LazyService",
    "ServiceError",
    "ServiceInitializationError",
    "ServiceNotReadyError",
]
