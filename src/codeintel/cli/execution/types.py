"""Unified type definitions for CLI execution pipeline.

Provide type aliases and dataclasses for handlers, progress events,
and streaming results that work across sync and async execution modes.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from codeintel.cli.results import CliResult

T = TypeVar("T")


class ProgressState(Enum):
    """Progress state for operations.

    Values
    ------
    PENDING
        Operation not yet started.
    RUNNING
        Operation in progress.
    PAUSED
        Operation temporarily paused.
    COMPLETED
        Operation finished successfully.
    FAILED
        Operation finished with error.
    CANCELLED
        Operation was cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProgressEvent:
    """Progress update from an operation.

    Works for both sync (via callbacks) and async (via generators).

    Parameters
    ----------
    operation_id
        Operation identifier.
    state
        Current progress state.
    progress
        Progress percentage (0.0 to 1.0), None if indeterminate.
    message
        Human-readable status message.
    details
        Optional additional details.
    timestamp
        Event timestamp.
    items_completed
        Number of items completed (optional).
    items_total
        Total number of items (optional).
    """

    operation_id: str
    state: ProgressState
    progress: float | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    items_completed: int | None = None
    items_total: int | None = None

    @property
    def percentage(self) -> float:
        """Get progress percentage.

        Returns
        -------
        float
            Progress as percentage (0-100), or 0 if indeterminate.
        """
        if self.progress is None:
            return 0.0
        return self.progress * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "operation_id": self.operation_id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "items_completed": self.items_completed,
            "items_total": self.items_total,
            "percentage": self.percentage,
        }


@dataclass
class StreamingResult[T]:
    """Result container for streaming operations.

    Encapsulate either progress events or the final result.

    Parameters
    ----------
    progress
        Progress event if this is a progress update.
    result
        Final result if this is the completion.
    """

    progress: ProgressEvent | None = None
    result: CliResult[T] | None = None

    @property
    def is_progress(self) -> bool:
        """Check if this is a progress update.

        Returns
        -------
        bool
            True if this contains a progress event.
        """
        return self.progress is not None

    @property
    def is_result(self) -> bool:
        """Check if this is a final result.

        Returns
        -------
        bool
            True if this contains a final result.
        """
        return self.result is not None


@dataclass
class ProgressConfig:
    """Configuration for progress reporting.

    Parameters
    ----------
    enabled
        Whether progress reporting is enabled.
    show_spinner
        Whether to show spinner for indeterminate progress.
    update_interval
        Minimum interval between updates in seconds.
    format_string
        Format string for progress display.
    """

    enabled: bool = True
    show_spinner: bool = True
    update_interval: float = 0.1
    format_string: str = "{message} [{current}/{total}]"


# Type aliases for handlers
# Note: These are defined as string annotations to avoid runtime evaluation issues
# with generic type subscripting. Use them in type hints, not as runtime values.
type SyncHandler[T] = Callable[..., CliResult[T]]
"""Synchronous handler type."""

type AsyncHandler[T] = Callable[..., Awaitable[CliResult[T]]]
"""Asynchronous handler type."""

type StreamingHandler[T] = Callable[..., AsyncGenerator[StreamingResult[T]]]
"""Streaming handler that yields progress events and final result."""

type AnyHandler[T] = SyncHandler[T] | AsyncHandler[T] | StreamingHandler[T]
"""Union of all handler types."""


def is_async_handler(handler: object) -> bool:
    """Check if a handler is async.

    Parameters
    ----------
    handler
        Handler function to check.

    Returns
    -------
    bool
        True if handler is async.
    """
    if asyncio.iscoroutinefunction(handler):
        return True

    # For callable objects, check if their type's __call__ is async
    if callable(handler):
        call_method = type(handler).__call__
        return asyncio.iscoroutinefunction(call_method)

    return False


def is_streaming_handler(handler: object) -> bool:
    """Check if a handler is a streaming (async generator) handler.

    Parameters
    ----------
    handler
        Handler function to check.

    Returns
    -------
    bool
        True if handler is an async generator.
    """
    if inspect.isasyncgenfunction(handler):
        return True

    # For callable objects, check if their type's __call__ is an async generator
    if callable(handler):
        call_method = type(handler).__call__
        return inspect.isasyncgenfunction(call_method)

    return False


def get_handler_type(handler: object) -> str:
    """Determine the type of a handler.

    Parameters
    ----------
    handler
        Handler function to check.

    Returns
    -------
    str
        One of 'sync', 'async', or 'streaming'.
    """
    if is_streaming_handler(handler):
        return "streaming"
    if is_async_handler(handler):
        return "async"
    return "sync"


__all__ = [
    "AnyHandler",
    "AsyncHandler",
    "ProgressConfig",
    "ProgressEvent",
    "ProgressState",
    "StreamingHandler",
    "StreamingResult",
    "SyncHandler",
    "get_handler_type",
    "is_async_handler",
    "is_streaming_handler",
]
