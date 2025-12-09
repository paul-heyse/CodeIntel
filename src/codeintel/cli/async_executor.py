"""Async-aware operation executor.

Extend the standard OperationExecutor to support async handlers,
streaming progress, and proper cancellation propagation.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from codeintel.cli.async_types import (
    ProgressEvent,
    ProgressState,
    StreamingResult,
    get_handler_type,
)
from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.executor import ExecutionResult, OperationExecutor, OperationSpec, get_executor
from codeintel.cli.results import CliResult


@dataclass
class AsyncExecutionContext:
    """Context for async operation execution.

    Provide cancellation support and progress tracking for async operations.

    Parameters
    ----------
    operation_id
        Operation identifier.
    params
        Operation parameters.
    cancel_event
        Event to signal cancellation.
    progress_callback
        Optional callback for progress updates.
    """

    operation_id: str
    params: dict[str, Any]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    progress_callback: Any = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled.

        Returns
        -------
        bool
            True if cancellation was requested.
        """
        return self.cancel_event.is_set()

    def request_cancellation(self) -> None:
        """Request operation cancellation."""
        self.cancel_event.set()

    async def check_cancelled(self) -> None:
        """Raise CancelledError if cancellation was requested.

        Raises
        ------
        asyncio.CancelledError
            If cancellation was requested.
        """
        if self.is_cancelled:
            msg = f"Operation {self.operation_id} was cancelled"
            raise asyncio.CancelledError(msg)

    def report_progress(
        self,
        progress: float | None = None,
        message: str = "",
        *,
        items_completed: int | None = None,
        items_total: int | None = None,
    ) -> ProgressEvent:
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
        ProgressEvent
            The created progress event.
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


@dataclass
class AsyncExecutionResult[T]:
    """Result of async operation execution.

    Parameters
    ----------
    result
        The CLI result.
    duration_ms
        Execution duration in milliseconds.
    was_cancelled
        Whether operation was cancelled.
    progress_events
        List of progress events emitted.
    """

    result: CliResult[T]
    duration_ms: float
    was_cancelled: bool = False
    progress_events: list[ProgressEvent] = field(default_factory=list)


class AsyncOperationExecutor:
    """Execute CLI operations with async support.

    Extend the standard executor to handle async handlers,
    progress streaming, and cancellation.

    Parameters
    ----------
    executor
        Underlying sync executor (default: global executor).
    """

    def __init__(self, executor: OperationExecutor | None = None) -> None:
        """Initialize async executor."""
        self._executor = executor or get_executor()

    async def execute_async[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        render: bool = True,
        timeout_seconds: float | None = None,
    ) -> AsyncExecutionResult[T]:
        """Execute an operation asynchronously.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        render
            Whether to render output.
        timeout_seconds
            Operation timeout in seconds.

        Returns
        -------
        AsyncExecutionResult[T]
            Execution result with async metadata.
        """
        start = datetime.now(UTC)
        handler_type = get_handler_type(spec.handler)
        was_cancelled = False

        try:
            if handler_type == "async":
                # Native async handler
                if timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        self._execute_async_handler(spec, params),
                        timeout=timeout_seconds,
                    )
                else:
                    result = await self._execute_async_handler(spec, params)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                if timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._execute_sync_handler(spec, params, render=render),
                        ),
                        timeout=timeout_seconds,
                    )
                else:
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._execute_sync_handler(spec, params, render=render),
                    )

        except asyncio.CancelledError:
            was_cancelled = True
            result = CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:operation/cancelled",
                    title="Operation Cancelled",
                    detail=f"Operation {spec.operation_id} was cancelled",
                    status=499,
                ),
            )
        except TimeoutError:
            result = CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:operation/timeout",
                    title="Operation Timeout",
                    detail=f"Operation {spec.operation_id} exceeded timeout",
                    status=504,
                ),
            )

        end = datetime.now(UTC)
        duration_ms = (end - start).total_seconds() * 1000

        return AsyncExecutionResult(
            result=result,
            duration_ms=duration_ms,
            was_cancelled=was_cancelled,
        )

    @staticmethod
    async def _execute_async_handler[T](
        spec: OperationSpec[T],
        params: dict[str, Any],
    ) -> CliResult[T]:
        """Execute an async handler.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        handler = spec.handler
        return await handler(**params)  # type: ignore[misc]

    def _execute_sync_handler[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        render: bool,
    ) -> CliResult[T]:
        """Execute a sync handler.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        render
            Whether to render output.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        result: ExecutionResult[T] = self._executor.execute(spec, params, render=render)
        return result.result

    async def execute_streaming[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[StreamingResult[T]]:
        """Execute an operation with streaming progress.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        cancel_event
            Event to signal cancellation.

        Yields
        ------
        StreamingResult[T]
            Progress events and final result.

        Raises
        ------
        asyncio.CancelledError
            If operation is cancelled.
        """
        cancel_event = cancel_event or asyncio.Event()

        # Initial progress event
        yield StreamingResult[T](
            progress=ProgressEvent(
                operation_id=spec.operation_id,
                state=ProgressState.RUNNING,
                message=f"Starting {spec.operation_id}",
            ),
        )

        def _check_cancellation() -> None:
            """Raise CancelledError if cancellation requested.

            Raises
            ------
            asyncio.CancelledError
                If cancellation has been requested.
            """
            if cancel_event.is_set():
                raise asyncio.CancelledError

        try:
            # Check for cancellation
            _check_cancellation()

            # Execute the operation
            result = await self.execute_async(spec, params, render=False)

            # Completion event
            state = ProgressState.COMPLETED if result.result.success else ProgressState.FAILED
            yield StreamingResult[T](
                progress=ProgressEvent(
                    operation_id=spec.operation_id,
                    state=state,
                    progress=1.0 if result.result.success else None,
                    message="Completed" if result.result.success else "Failed",
                ),
            )

            # Final result
            yield StreamingResult[T](result=result.result)

        except asyncio.CancelledError:
            yield StreamingResult[T](
                progress=ProgressEvent(
                    operation_id=spec.operation_id,
                    state=ProgressState.CANCELLED,
                    message="Operation cancelled",
                ),
            )
            raise


@asynccontextmanager
async def cancellable_operation(
    operation_id: str,
    params: dict[str, Any],
) -> AsyncGenerator[AsyncExecutionContext]:
    """Create a cancellable operation context.

    Parameters
    ----------
    operation_id
        Operation identifier.
    params
        Operation parameters.

    Yields
    ------
    AsyncExecutionContext
        Context for the operation.

    Raises
    ------
    asyncio.CancelledError
        If operation is cancelled.

    Examples
    --------
    >>> async with cancellable_operation("my.op", {}) as ctx:
    ...     ctx.report_progress(0.5, "Half done")
    ...     await ctx.check_cancelled()
    """
    ctx = AsyncExecutionContext(operation_id=operation_id, params=params)
    try:
        # Yield in async context for potential async operations in the block
        await asyncio.sleep(0)  # Allow async context switch
        yield ctx
    except asyncio.CancelledError:
        ctx.request_cancellation()
        raise


async def run_async_operation[T](
    spec: OperationSpec[T],
    params: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> CliResult[T]:
    """Run an operation asynchronously.

    Convenience function for running a single async operation.

    Parameters
    ----------
    spec
        Operation specification.
    params
        Operation parameters.
    timeout_seconds
        Optional timeout in seconds.

    Returns
    -------
    CliResult[T]
        Operation result.
    """
    executor = AsyncOperationExecutor()
    result = await executor.execute_async(spec, params, timeout_seconds=timeout_seconds)
    return result.result


def run_sync[T](
    spec: OperationSpec[T],
    params: dict[str, Any],
) -> CliResult[T]:
    """Run an async-capable operation synchronously.

    Parameters
    ----------
    spec
        Operation specification.
    params
        Operation parameters.

    Returns
    -------
    CliResult[T]
        Operation result.
    """
    handler_type = get_handler_type(spec.handler)

    if handler_type == "sync":
        # Direct sync execution
        executor = get_executor()
        result = executor.execute(spec, params, render=False)
        return result.result

    # Run async handler in event loop
    with contextlib.suppress(RuntimeError):
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We're already in an async context, use nest_asyncio pattern
            # or just run the sync handler
            executor = get_executor()
            result = executor.execute(spec, params, render=False)
            return result.result

    # Create new event loop and run
    return asyncio.run(run_async_operation(spec, params))


__all__ = [
    "AsyncExecutionContext",
    "AsyncExecutionResult",
    "AsyncOperationExecutor",
    "cancellable_operation",
    "run_async_operation",
    "run_sync",
]
