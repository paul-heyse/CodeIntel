"""Centralized error definitions for plugin execution.

This module provides common error types and exception tuples used across
all plugin execution domains (graphs, ingestion, analytics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.helpers import DUCKDB_ERRORS

if TYPE_CHECKING:
    from codeintel.core.plugins.types.result import PluginExecutionRecord

# Exceptions that can be caught and handled during plugin execution.
# These represent recoverable errors that should not crash the entire run.
PLUGIN_CATCHABLE_ERRORS: tuple[type[Exception], ...] = (
    *DUCKDB_ERRORS,
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


class PluginFatalError(Exception):
    """Fatal plugin failure while respecting fail-fast semantics.

    Raise this exception when a plugin failure should terminate the entire
    execution run due to fail-fast semantics being enabled. The exception
    captures both the execution record at the time of failure and the
    original exception that caused the failure.

    Attributes
    ----------
    record
        The execution record at time of failure.
    original
        The exception that caused the failure.

    Examples
    --------
    >>> record = PluginExecutionRecord(...)
    >>> try:
    ...     # Plugin execution that fails
    ...     raise ValueError("Something went wrong")
    ... except ValueError as e:
    ...     raise PluginFatalError(record, e) from e
    """

    def __init__(
        self,
        record: PluginExecutionRecord,
        original: Exception,
    ) -> None:
        """Initialize with execution record and original exception.

        Parameters
        ----------
        record
            The execution record at time of failure.
        original
            The exception that caused the failure.
        """
        super().__init__(str(original))
        self.record = record
        self.original = original


class PluginTimeoutError(Exception):
    """Plugin execution exceeded the configured timeout.

    Attributes
    ----------
    plugin_name
        Name of the plugin that timed out.
    timeout_ms
        The configured timeout in milliseconds.
    elapsed_ms
        Actual elapsed time before timeout.
    """

    def __init__(
        self,
        plugin_name: str,
        timeout_ms: int,
        elapsed_ms: float | None = None,
    ) -> None:
        """Initialize with plugin name and timeout details.

        Parameters
        ----------
        plugin_name
            Name of the plugin that timed out.
        timeout_ms
            The configured timeout in milliseconds.
        elapsed_ms
            Actual elapsed time before timeout.
        """
        message = f"Plugin '{plugin_name}' exceeded timeout of {timeout_ms}ms"
        if elapsed_ms is not None:
            message += f" (elapsed: {elapsed_ms:.2f}ms)"
        super().__init__(message)
        self.plugin_name = plugin_name
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms


class PluginSkippedError(Exception):
    """Plugin was skipped due to configuration or dependency issues.

    This is not a true error but used to signal skip conditions in
    control flow where exceptions are the cleanest pattern.

    Attributes
    ----------
    plugin_name
        Name of the plugin that was skipped.
    reason
        Reason for skipping.
    """

    def __init__(self, plugin_name: str, reason: str) -> None:
        """Initialize with plugin name and skip reason.

        Parameters
        ----------
        plugin_name
            Name of the plugin that was skipped.
        reason
            Reason for skipping.
        """
        super().__init__(f"Plugin '{plugin_name}' skipped: {reason}")
        self.plugin_name = plugin_name
        self.reason = reason


class PluginSkipRequestError(Exception):
    """Internal signal to request a plugin skip execution.

    This is not a true error condition - it signals that a plugin has determined
    it should skip processing (e.g., no matching files, already processed).

    Unlike PluginSkippedError, this is used for internal control flow within
    a plugin's execute() method, not for external skip decisions made during
    planning.

    Examples
    --------
    >>> def execute(self, ctx):
    ...     if not self._has_work_to_do(ctx):
    ...         raise PluginSkipRequestError("No files to process")
    ...     # ... do work ...
    """

    def __init__(self, reason: str = "") -> None:
        """Initialize with optional skip reason.

        Parameters
        ----------
        reason
            Optional description of why execution is being skipped.
        """
        super().__init__(reason)
        self.reason = reason


__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "PluginFatalError",
    "PluginSkipRequestError",
    "PluginSkippedError",
    "PluginTimeoutError",
]
