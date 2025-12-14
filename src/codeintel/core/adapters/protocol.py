"""Adapter protocol definitions.

This module defines the core protocols for adapters in a hexagonal
architecture, providing a standardized interface for connecting
to external systems.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class AdapterProtocol(Protocol):
    """Protocol for adapters in hexagonal architecture.

    Adapters connect the application core to external systems
    (databases, APIs, file systems, etc.) through a consistent interface.

    Attributes
    ----------
    ADAPTER_NAME
        Unique identifier for this adapter type.

    Examples
    --------
    >>> class FileStorageAdapter:
    ...     ADAPTER_NAME: ClassVar[str] = "file_storage"
    ...
    ...     def initialize(self) -> None:
    ...         self._ensure_directories()
    ...
    ...     def close(self) -> None:
    ...         pass
    ...
    ...     @property
    ...     def is_available(self) -> bool:
    ...         return self._base_path.exists()
    """

    ADAPTER_NAME: ClassVar[str]

    def initialize(self) -> None:
        """Initialize the adapter.

        Perform any setup required before the adapter can be used.
        This may include connecting to services, creating directories, etc.
        """
        ...

    def close(self) -> None:
        """Close the adapter and release resources.

        Clean up any resources held by the adapter.
        Safe to call multiple times.
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the adapter is available for use.

        Returns
        -------
        bool
            True if the adapter is ready to handle requests.
        """
        ...


@runtime_checkable
class PortProtocol(Protocol):
    """Protocol for ports in hexagonal architecture.

    Ports define the interface that the application core expects.
    Adapters implement these ports to provide concrete implementations.

    This is a marker protocol - specific ports define their own methods.
    """


class AdapterError(Exception):
    """Base exception for adapter-related errors.

    Attributes
    ----------
    adapter_name
        Name of the adapter that raised the error.
    """

    def __init__(self, adapter_name: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        adapter_name
            Name of the adapter.
        message
            Error message.
        """
        super().__init__(f"[{adapter_name}] {message}")
        self.adapter_name = adapter_name


class AdapterNotAvailableError(AdapterError):
    """Raised when an adapter is not available."""


class AdapterInitializationError(AdapterError):
    """Raised when adapter initialization fails."""


__all__ = [
    "AdapterError",
    "AdapterInitializationError",
    "AdapterNotAvailableError",
    "AdapterProtocol",
    "PortProtocol",
]
