"""Base adapter implementation.

This module provides a base class for adapters with common patterns
for lifecycle management and error handling.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Self

from codeintel.core.adapters.protocol import (
    AdapterError,
    AdapterInitializationError,
)

log = logging.getLogger(__name__)


class BaseAdapter:
    """Base class for adapter implementations.

    Provides common patterns for adapter lifecycle management
    including initialization, availability checking, and cleanup.

    Subclasses should override `_do_initialize()` and `_do_close()`
    to implement their specific logic.

    Attributes
    ----------
    ADAPTER_NAME
        Unique identifier for this adapter type. Must be overridden.

    Examples
    --------
    >>> class DatabaseAdapter(BaseAdapter):
    ...     ADAPTER_NAME: ClassVar[str] = "database"
    ...
    ...     def _do_initialize(self) -> None:
    ...         self._connection = connect()
    ...
    ...     def _do_close(self) -> None:
    ...         self._connection.close()
    ...
    ...     @property
    ...     def is_available(self) -> bool:
    ...         return self._connection is not None
    """

    ADAPTER_NAME: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize the base adapter."""
        self._initialized = False
        self._closed = False
        self._error: Exception | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if adapter has been initialized.

        Returns
        -------
        bool
            True if initialized.
        """
        return self._initialized

    @property
    def is_closed(self) -> bool:
        """Check if adapter has been closed.

        Returns
        -------
        bool
            True if closed.
        """
        return self._closed

    @property
    def is_available(self) -> bool:
        """Check if adapter is available for use.

        Returns
        -------
        bool
            True if initialized and not closed.
        """
        return self._initialized and not self._closed

    @property
    def last_error(self) -> Exception | None:
        """Return the last error that occurred.

        Returns
        -------
        Exception | None
            Last error, or None.
        """
        return self._error

    def initialize(self) -> None:
        """Initialize the adapter.

        Raises
        ------
        AdapterInitializationError
            If initialization fails.
        """
        if self._initialized:
            log.debug("%s already initialized", self.ADAPTER_NAME)
            return

        if self._closed:
            msg = "Cannot initialize closed adapter"
            raise AdapterInitializationError(self.ADAPTER_NAME, msg)

        log.debug("Initializing %s", self.ADAPTER_NAME)

        try:
            self._do_initialize()
            self._initialized = True
            log.info("%s initialized successfully", self.ADAPTER_NAME)
        except Exception as e:
            self._error = e
            log.exception("Failed to initialize %s", self.ADAPTER_NAME)
            raise AdapterInitializationError(self.ADAPTER_NAME, str(e)) from e

    def close(self) -> None:
        """Close the adapter and release resources.

        Safe to call multiple times.
        """
        if self._closed:
            return

        log.debug("Closing %s", self.ADAPTER_NAME)

        try:
            self._do_close()
        except Exception:
            log.exception("Error during %s close", self.ADAPTER_NAME)
        finally:
            self._closed = True
            self._initialized = False
            log.info("%s closed", self.ADAPTER_NAME)

    def ensure_available(self) -> None:
        """Ensure the adapter is available.

        Raises
        ------
        AdapterError
            If the adapter is not available.
        """
        if not self.is_available:
            if self._closed:
                msg = "Adapter has been closed"
            elif not self._initialized:
                msg = "Adapter not initialized"
            else:
                msg = "Adapter not available"
            raise AdapterError(self.ADAPTER_NAME, msg)

    def _do_initialize(self) -> None:
        """Perform adapter-specific initialization.

        Subclasses override this to implement initialization logic.
        """

    def _do_close(self) -> None:
        """Perform adapter-specific cleanup.

        Subclasses override this to implement cleanup logic.
        """

    def __enter__(self) -> Self:
        """Enter context manager, initializing adapter.

        Returns
        -------
        Self
            The initialized adapter.
        """
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing adapter."""
        self.close()


__all__ = [
    "BaseAdapter",
]
