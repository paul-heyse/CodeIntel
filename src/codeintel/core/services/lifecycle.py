"""Service lifecycle management.

This module provides utilities for managing the lifecycle of multiple
services, including ordered startup and shutdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Self

from codeintel.core.services.base import ServiceError
from codeintel.core.services.protocol import ServiceProtocol, ServiceState

log = logging.getLogger(__name__)


class ServiceLifecycleError(ServiceError):
    """Raised when service lifecycle operations fail."""

    def __init__(self, message: str, failures: list[tuple[str, Exception]]) -> None:
        """Initialize the error.

        Parameters
        ----------
        message
            Error message.
        failures
            List of (service_name, exception) pairs for failed services.
        """
        super().__init__("lifecycle", message)
        self.failures = failures


@dataclass
class ServiceLifecycle:
    """Manage lifecycle of multiple services.

    Provides ordered initialization and shutdown of services,
    with proper error handling and logging.

    Examples
    --------
    >>> lifecycle = ServiceLifecycle()
    >>> lifecycle.register(database_service, priority=1)
    >>> lifecycle.register(cache_service, priority=2)
    >>> lifecycle.register(api_service, priority=3)
    >>> lifecycle.start_all()
    >>> # ... application runs ...
    >>> lifecycle.stop_all()
    """

    _services: list[tuple[int, ServiceProtocol]] = field(default_factory=list)
    _started: list[ServiceProtocol] = field(default_factory=list)

    def register(self, service: ServiceProtocol, *, priority: int = 0) -> None:
        """Register a service with the lifecycle manager.

        Services with lower priority values are started first
        and stopped last.

        Parameters
        ----------
        service
            Service to register.
        priority
            Startup priority (lower = earlier start, later stop).
        """
        self._services.append((priority, service))
        self._services.sort(key=lambda x: x[0])
        log.debug(
            "Registered service %s with priority %d",
            service.SERVICE_NAME,
            priority,
        )

    def unregister(self, service: ServiceProtocol) -> bool:
        """Remove a service from lifecycle management.

        Parameters
        ----------
        service
            Service to unregister.

        Returns
        -------
        bool
            True if service was found and removed.
        """
        found_index: int | None = None
        for i, (_, s) in enumerate(self._services):
            if s is service:
                found_index = i
                break

        if found_index is not None:
            self._services.pop(found_index)
            if service in self._started:
                self._started.remove(service)
            log.debug("Unregistered service %s", service.SERVICE_NAME)
            return True
        return False

    def start_all(self, *, stop_on_failure: bool = True) -> None:
        """Start all registered services in priority order.

        Parameters
        ----------
        stop_on_failure
            If True, stop already-started services when one fails.

        Raises
        ------
        ServiceLifecycleError
            If any service fails to start.
        """
        failures: list[tuple[str, Exception]] = []

        for _, service in self._services:
            try:
                service.initialize()
                self._started.append(service)
                log.info("Started service: %s", service.SERVICE_NAME)
            except Exception as e:
                log.exception("Failed to start service: %s", service.SERVICE_NAME)
                failures.append((service.SERVICE_NAME, e))

                if stop_on_failure:
                    self._stop_started_services()
                    msg = f"Failed to start {service.SERVICE_NAME}"
                    raise ServiceLifecycleError(msg, failures) from e

        if failures:
            msg = f"Failed to start {len(failures)} service(s)"
            raise ServiceLifecycleError(msg, failures)

    def stop_all(self) -> list[tuple[str, Exception]]:
        """Stop all started services in reverse priority order.

        Returns
        -------
        list[tuple[str, Exception]]
            List of (service_name, exception) pairs for failed shutdowns.
        """
        return self._stop_started_services()

    def _stop_started_services(self) -> list[tuple[str, Exception]]:
        """Stop services in reverse order they were started.

        Returns
        -------
        list[tuple[str, Exception]]
            List of failures during shutdown.
        """
        failures: list[tuple[str, Exception]] = []

        for service in reversed(self._started):
            try:
                service.shutdown()
                log.info("Stopped service: %s", service.SERVICE_NAME)
            except Exception as e:
                log.exception("Failed to stop service: %s", service.SERVICE_NAME)
                failures.append((service.SERVICE_NAME, e))

        self._started.clear()
        return failures

    def restart_all(self) -> None:
        """Restart all services.

        Stops all services and then starts them again.
        May raise exceptions from start_all().
        """
        self.stop_all()
        self.start_all()

    def restart(self, service: ServiceProtocol) -> None:
        """Restart a specific service.

        Parameters
        ----------
        service
            Service to restart.
        """
        if service in self._started:
            service.shutdown()
            self._started.remove(service)

        service.initialize()
        self._started.append(service)
        log.info("Restarted service: %s", service.SERVICE_NAME)

    @property
    def all_ready(self) -> bool:
        """Check if all services are ready.

        Returns
        -------
        bool
            True if all registered services are ready.
        """
        return all(s.is_ready for _, s in self._services)

    @property
    def service_states(self) -> dict[str, ServiceState]:
        """Get states of all registered services.

        Returns
        -------
        dict[str, ServiceState]
            Mapping of service name to state.
        """
        result: dict[str, ServiceState] = {}
        for _, service in self._services:
            state = getattr(service, "state", None)
            if isinstance(state, ServiceState):
                result[service.SERVICE_NAME] = state
            elif service.is_ready:
                result[service.SERVICE_NAME] = ServiceState.READY
            else:
                result[service.SERVICE_NAME] = ServiceState.CREATED
        return result

    @property
    def registered_services(self) -> tuple[str, ...]:
        """Get names of all registered services.

        Returns
        -------
        tuple[str, ...]
            Service names in priority order.
        """
        return tuple(s.SERVICE_NAME for _, s in self._services)

    def __len__(self) -> int:
        """Return number of registered services.

        Returns
        -------
        int
            Count of registered services.
        """
        return len(self._services)

    def __enter__(self) -> Self:
        """Enter context manager, starting all services.

        Returns
        -------
        Self
            Self with all services started.
        """
        self.start_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, stopping all services."""
        self.stop_all()


__all__ = [
    "ServiceLifecycle",
    "ServiceLifecycleError",
]
