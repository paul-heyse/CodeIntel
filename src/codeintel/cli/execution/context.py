"""Unified execution context for CLI operations.

Provide context objects that track operation state for both
sync and async execution modes, with lazy resolution of runtime
and gateway resources.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.cli.execution.types import ProgressEvent, ProgressState
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.core import CliResult
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway


@dataclass
class ExecutionContext:
    """Context passed through the execution pipeline.

    Work for both sync and async operations with optional
    cancellation and progress support. Provides lazy resolution
    of runtime and gateway resources via require_runtime() and
    require_gateway() methods.

    Parameters
    ----------
    operation_id
        The operation being executed.
    params
        Validated operation parameters.
    output_format
        Requested output format.
    start_time
        Execution start timestamp (monotonic).
    started_at
        Execution start datetime.
    metadata
        Additional context metadata.
    cancellation_event
        Event to signal cancellation (async only).
    progress_callback
        Optional callback for progress updates.

    Examples
    --------
    >>> ctx = ExecutionContext.for_sync("build.run", {"targets": ["all"]})
    >>> runtime = ctx.require_runtime()  # doctest: +SKIP
    >>> gateway = ctx.require_gateway()  # doctest: +SKIP
    >>> ctx.logger.info("Building targets")  # doctest: +SKIP
    """

    operation_id: str
    params: dict[str, Any]
    output_format: OutputFormat = OutputFormat.TEXT
    start_time: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    cancellation_event: asyncio.Event | None = None
    progress_callback: Callable[[ProgressEvent], None] | None = None

    # Private lazy-resolved fields (not in repr)
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _gateway_read_only: bool = field(default=True, repr=False)

    # --- Lazy Resource Resolution ---

    def require_runtime(self) -> ResolvedRuntime:
        """Get resolved runtime, resolving lazily if needed.

        Resolution is cached after first call. Uses the params dict
        to resolve the project runtime via project file discovery or
        explicit CLI parameters.

        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime information.

        Notes
        -----
        Propagates ResolutionError from resolve_runtime if runtime cannot
        be resolved from params (no project file and missing repo/commit).
        """
        if self._runtime is None:
            self._runtime = _lazy_resolve_runtime(self)
        return self._runtime

    def require_gateway(self, *, read_only: bool = True) -> StorageGateway:
        """Get gateway, opening lazily if needed.

        Gateway is cached after first call. If the gateway was previously
        opened with a different read_only mode, it is closed and reopened.

        Parameters
        ----------
        read_only
            Whether to open in read-only mode.

        Returns
        -------
        StorageGateway
            Open gateway.

        Notes
        -----
        Propagates ResolutionError from require_runtime if runtime cannot
        be resolved, or StorageConnectionError from open_gateway if the
        database cannot be opened.
        """
        # Reopen if read_only mode changed
        if self._gateway is not None and self._gateway_read_only != read_only:
            self._gateway.close()
            self._gateway = None

        if self._gateway is None:
            self._gateway = _lazy_open_gateway(self, read_only=read_only)
            self._gateway_read_only = read_only

        return self._gateway

    def close(self) -> None:
        """Close any open resources.

        Should be called when execution completes, typically by the executor.
        Safe to call multiple times.
        """
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    # --- Parameter Access ---

    def get_str_param(self, key: str, default: str | None = None) -> str | None:
        """Get string parameter with default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Value if parameter not present. Defaults to None.

        Returns
        -------
        str | None
            Parameter value or default.
        """
        value = self.params.get(key)
        return str(value) if value is not None else default

    def get_int_param(self, key: str, default: int) -> int:
        """Get integer parameter with default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Value if parameter not present.

        Returns
        -------
        int
            Parameter value or default.
        """
        value = self.params.get(key)
        return int(value) if value is not None else default

    def get_bool_param(self, key: str, *, default: bool = False) -> bool:
        """Get boolean parameter with default.

        Parameters
        ----------
        key
            Parameter name.
        default
            Value if parameter not present (keyword-only).

        Returns
        -------
        bool
            Parameter value or default.
        """
        value = self.params.get(key)
        return bool(value) if value is not None else default

    def require_str_param(self, key: str) -> str:
        """Get required string parameter.

        Parameters
        ----------
        key
            Parameter name.

        Returns
        -------
        str
            Parameter value.

        Raises
        ------
        ValueError
            If parameter is missing.
        """
        value = self.params.get(key)
        if value is None:
            msg = f"Required parameter '{key}' not provided"
            raise ValueError(msg)
        return str(value)

    # --- Convenience Properties ---

    @property
    def verbosity(self) -> int:
        """Get verbosity level from params.

        Returns
        -------
        int
            Verbosity level (0-2+).
        """
        return self.params.get("verbose", 0)

    @property
    def dry_run(self) -> bool:
        """Check if this is a dry-run.

        Returns
        -------
        bool
            True if dry-run mode.
        """
        return self.params.get("dry_run", False)

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this operation.

        Returns
        -------
        logging.Logger
            Logger named for this operation.
        """
        return logging.getLogger(f"codeintel.cli.{self.operation_id}")

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed execution time.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        return time.monotonic() - self.start_time

    @property
    def is_async(self) -> bool:
        """Check if this is an async execution context.

        Returns
        -------
        bool
            True if cancellation support is enabled.
        """
        return self.cancellation_event is not None

    @property
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled.

        Returns
        -------
        bool
            True if cancellation was requested.
        """
        if self.cancellation_event is None:
            return False
        return self.cancellation_event.is_set()

    # --- Cancellation Support ---

    def check_cancelled(self) -> None:
        """Check if operation was cancelled and raise if so.

        Raises
        ------
        asyncio.CancelledError
            If cancellation was requested.
        """
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            msg = f"Operation {self.operation_id} was cancelled"
            raise asyncio.CancelledError(msg)

    async def check_cancelled_async(self) -> None:
        """Async version of cancellation check.

        Allow event loop to process other tasks.
        Delegates to check_cancelled() which may raise CancelledError.
        """
        self.check_cancelled()
        await asyncio.sleep(0)

    def request_cancellation(self) -> None:
        """Request operation cancellation."""
        if self.cancellation_event is not None:
            self.cancellation_event.set()

    # --- Progress Reporting ---

    def report_progress(
        self,
        progress: float | None = None,
        message: str = "",
        *,
        items_completed: int | None = None,
        items_total: int | None = None,
    ) -> ProgressEvent | None:
        """Create and optionally report a progress event.

        Parameters
        ----------
        progress
            Progress percentage (0.0 to 1.0).
        message
            Status message.
        items_completed
            Items completed count.
        items_total
            Total items count.

        Returns
        -------
        ProgressEvent | None
            The created progress event, or None if no callback.
        """
        event = ProgressEvent(
            operation_id=self.operation_id,
            state=ProgressState.RUNNING,
            progress=progress,
            message=message,
            items_completed=items_completed,
            items_total=items_total,
        )

        if self.progress_callback is not None:
            self.progress_callback(event)

        return event

    # --- Factory Methods ---

    @classmethod
    def for_sync(
        cls,
        operation_id: str,
        params: dict[str, Any],
        output_format: OutputFormat = OutputFormat.TEXT,
    ) -> ExecutionContext:
        """Create context for sync execution.

        Parameters
        ----------
        operation_id
            Operation identifier.
        params
            Operation parameters.
        output_format
            Output format.

        Returns
        -------
        ExecutionContext
            Context for sync execution.
        """
        return cls(
            operation_id=operation_id,
            params=params,
            output_format=output_format,
        )

    @classmethod
    def for_async(
        cls,
        operation_id: str,
        params: dict[str, Any],
        output_format: OutputFormat = OutputFormat.TEXT,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> ExecutionContext:
        """Create context for async execution.

        Parameters
        ----------
        operation_id
            Operation identifier.
        params
            Operation parameters.
        output_format
            Output format.
        progress_callback
            Optional progress callback.

        Returns
        -------
        ExecutionContext
            Context for async execution with cancellation support.
        """
        return cls(
            operation_id=operation_id,
            params=params,
            output_format=output_format,
            cancellation_event=asyncio.Event(),
            progress_callback=progress_callback,
        )


@dataclass
class ExecutionResult[T]:
    """Result of operation execution with metrics.

    Parameters
    ----------
    result
        The CliResult from the handler.
    duration_seconds
        Total execution duration.
    validation_errors
        Any validation errors encountered.
    retries
        Number of retry attempts.
    was_cancelled
        Whether operation was cancelled.
    progress_events
        Progress events emitted during execution.
    """

    result: CliResult[T]
    duration_seconds: float
    validation_errors: list[str] = field(default_factory=list)
    retries: int = 0
    was_cancelled: bool = False
    progress_events: list[ProgressEvent] = field(default_factory=list)


def _lazy_resolve_runtime(ctx: ExecutionContext) -> ResolvedRuntime:
    """Resolve runtime, importing lazily to avoid circular imports.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    ResolvedRuntime
        Resolved runtime.
    """
    from codeintel.cli.resolution import resolve_runtime  # noqa: PLC0415

    return resolve_runtime(ctx)


def _lazy_open_gateway(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open gateway, importing lazily to avoid circular imports.

    Parameters
    ----------
    ctx
        Execution context.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    from codeintel.cli.resolution import open_gateway_for_context  # noqa: PLC0415

    return open_gateway_for_context(ctx, read_only=read_only)


__all__ = [
    "ExecutionContext",
    "ExecutionResult",
]
