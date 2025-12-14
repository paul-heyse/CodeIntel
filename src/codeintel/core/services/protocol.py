"""Service protocol definitions.

This module defines the core protocols for service implementations,
providing a standardized interface for service lifecycle management.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Protocol, runtime_checkable


class ServiceState(Enum):
    """Service lifecycle states.

    Attributes
    ----------
    CREATED
        Service has been instantiated but not initialized.
    INITIALIZING
        Service is currently initializing.
    READY
        Service is initialized and ready to handle requests.
    SHUTTING_DOWN
        Service is in the process of shutting down.
    STOPPED
        Service has been stopped and cannot handle requests.
    FAILED
        Service initialization or operation failed.
    """

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for all service types.

    Services are long-lived components that provide specific functionality
    to the application. They follow a consistent lifecycle pattern with
    initialization and shutdown phases.

    Attributes
    ----------
    SERVICE_NAME
        Unique identifier for the service type.

    Examples
    --------
    >>> class MyService:
    ...     SERVICE_NAME: ClassVar[str] = "my_service"
    ...
    ...     def initialize(self) -> None:
    ...         # Setup resources
    ...         pass
    ...
    ...     def shutdown(self) -> None:
    ...         # Cleanup resources
    ...         pass
    ...
    ...     @property
    ...     def is_ready(self) -> bool:
    ...         return True
    """

    SERVICE_NAME: ClassVar[str]

    def initialize(self) -> None:
        """Initialize the service.

        Perform any setup required before the service can handle requests.
        This may include connecting to databases, loading configuration,
        or initializing caches.

        Raises
        ------
        ServiceInitializationError
            If initialization fails.
        """
        ...

    def shutdown(self) -> None:
        """Shut down the service gracefully.

        Release resources, close connections, and perform cleanup.
        This method should be safe to call multiple times.
        """
        ...

    @property
    def is_ready(self) -> bool:
        """Check whether the service is ready to handle requests.

        Returns
        -------
        bool
            True if the service is initialized and ready.
        """
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for services that support health checks.

    Implement this protocol to provide detailed health information
    beyond the basic is_ready check.
    """

    def health_check(self) -> HealthStatus:
        """Perform a health check on the service.

        Returns
        -------
        HealthStatus
            Detailed health status of the service.
        """
        ...


class HealthStatus:
    """Health status of a service.

    Attributes
    ----------
    healthy
        Whether the service is healthy.
    message
        Optional message describing the health status.
    details
        Additional details about the health check.
    """

    def __init__(
        self,
        *,
        healthy: bool,
        message: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize health status.

        Parameters
        ----------
        healthy
            Whether the service is healthy.
        message
            Optional status message.
        details
            Additional health check details.
        """
        self.healthy = healthy
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            String representation of the health status.
        """
        status = "healthy" if self.healthy else "unhealthy"
        if self.message:
            return f"HealthStatus({status}: {self.message})"
        return f"HealthStatus({status})"


__all__ = [
    "HealthCheckProtocol",
    "HealthStatus",
    "ServiceProtocol",
    "ServiceState",
]
