"""Unified execution context for CLI operations.

Provide context objects that track operation state for both
sync and async execution modes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.execution.types import ProgressEvent, ProgressState

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.results import CliResult


@dataclass
class ExecutionContext:
    """Context passed through the execution pipeline.

    Work for both sync and async operations with optional
    cancellation and progress support.

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
    """

    operation_id: str
    params: dict[str, Any]
    output_format: OutputFormat = OutputFormat.TEXT
    start_time: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    cancellation_event: asyncio.Event | None = None
    progress_callback: Callable[[ProgressEvent], None] | None = None

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


__all__ = [
    "ExecutionContext",
    "ExecutionResult",
]
